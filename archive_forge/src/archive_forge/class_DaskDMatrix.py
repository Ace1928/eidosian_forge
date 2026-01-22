import collections
import logging
import platform
import socket
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, update_wrapper
from threading import Thread
from typing import (
import numpy
from . import collective, config
from ._typing import _T, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import DataFrame, LazyLoader, concat, lazy_isinstance
from .core import (
from .data import _is_cudf_ser, _is_cupy_array
from .sklearn import (
from .tracker import RabitTracker, get_host_ip
from .training import train as worker_train
class DaskDMatrix:
    """DMatrix holding on references to Dask DataFrame or Dask Array.  Constructing a
    `DaskDMatrix` forces all lazy computation to be carried out.  Wait for the input
    data explicitly if you want to see actual computation of constructing `DaskDMatrix`.

    See doc for :py:obj:`xgboost.DMatrix` constructor for other parameters.  DaskDMatrix
    accepts only dask collection.

    .. note::

        DaskDMatrix does not repartition or move data between workers.  It's
        the caller's responsibility to balance the data.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client :
        Specify the dask client used for training.  Use default client returned from
        dask if it's set to None.

    """

    @_deprecate_positional_args
    def __init__(self, client: 'distributed.Client', data: _DataT, label: Optional[_DaskCollection]=None, *, weight: Optional[_DaskCollection]=None, base_margin: Optional[_DaskCollection]=None, missing: Optional[float]=None, silent: bool=False, feature_names: Optional[FeatureNames]=None, feature_types: Optional[FeatureTypes]=None, group: Optional[_DaskCollection]=None, qid: Optional[_DaskCollection]=None, label_lower_bound: Optional[_DaskCollection]=None, label_upper_bound: Optional[_DaskCollection]=None, feature_weights: Optional[_DaskCollection]=None, enable_categorical: bool=False) -> None:
        _assert_dask_support()
        client = _xgb_get_client(client)
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.missing = missing if missing is not None else numpy.nan
        self.enable_categorical = enable_categorical
        if qid is not None and weight is not None:
            raise NotImplementedError('per-group weight is not implemented.')
        if group is not None:
            raise NotImplementedError('group structure is not implemented, use qid instead.')
        if len(data.shape) != 2:
            raise ValueError(f'Expecting 2 dimensional input, got: {data.shape}')
        if not isinstance(data, (dd.DataFrame, da.Array)):
            raise TypeError(_expect((dd.DataFrame, da.Array), type(data)))
        if not isinstance(label, (dd.DataFrame, da.Array, dd.Series, type(None))):
            raise TypeError(_expect((dd.DataFrame, da.Array, dd.Series), type(label)))
        self._n_cols = data.shape[1]
        assert isinstance(self._n_cols, int)
        self.worker_map: Dict[str, List[distributed.Future]] = defaultdict(list)
        self.is_quantile: bool = False
        self._init = client.sync(self._map_local_data, client, data, label=label, weights=weight, base_margin=base_margin, qid=qid, feature_weights=feature_weights, label_lower_bound=label_lower_bound, label_upper_bound=label_upper_bound)

    def __await__(self) -> Generator:
        return self._init.__await__()

    async def _map_local_data(self, client: 'distributed.Client', data: _DataT, label: Optional[_DaskCollection]=None, weights: Optional[_DaskCollection]=None, base_margin: Optional[_DaskCollection]=None, qid: Optional[_DaskCollection]=None, feature_weights: Optional[_DaskCollection]=None, label_lower_bound: Optional[_DaskCollection]=None, label_upper_bound: Optional[_DaskCollection]=None) -> 'DaskDMatrix':
        """Obtain references to local data."""
        from dask.delayed import Delayed

        def inconsistent(left: List[Any], left_name: str, right: List[Any], right_name: str) -> str:
            msg = f'Partitions between {left_name} and {right_name} are not consistent: {len(left)} != {len(right)}.  Please try to repartition/rechunk your data.'
            return msg

        def check_columns(parts: numpy.ndarray) -> None:
            assert parts.ndim == 1 or parts.shape[1], 'Data should be partitioned by row. To avoid this specify the number of columns for your dask Array explicitly. e.g. chunks=(partition_size, X.shape[1])'

        def to_delayed(d: _DaskCollection) -> List[Delayed]:
            """Breaking data into partitions, a trick borrowed from dask_xgboost. `to_delayed`
            downgrades high-level objects into numpy or pandas equivalents .

            """
            d = client.persist(d)
            delayed_obj = d.to_delayed()
            if isinstance(delayed_obj, numpy.ndarray):
                check_columns(delayed_obj)
                delayed_list: List[Delayed] = delayed_obj.flatten().tolist()
            else:
                delayed_list = delayed_obj
            return delayed_list

        def flatten_meta(meta: Optional[_DaskCollection]) -> Optional[List[Delayed]]:
            if meta is not None:
                meta_parts: List[Delayed] = to_delayed(meta)
                return meta_parts
            return None
        X_parts = to_delayed(data)
        y_parts = flatten_meta(label)
        w_parts = flatten_meta(weights)
        margin_parts = flatten_meta(base_margin)
        qid_parts = flatten_meta(qid)
        ll_parts = flatten_meta(label_lower_bound)
        lu_parts = flatten_meta(label_upper_bound)
        parts: Dict[str, List[Delayed]] = {'data': X_parts}

        def append_meta(m_parts: Optional[List[Delayed]], name: str) -> None:
            if m_parts is not None:
                assert len(X_parts) == len(m_parts), inconsistent(X_parts, 'X', m_parts, name)
                parts[name] = m_parts
        append_meta(y_parts, 'label')
        append_meta(w_parts, 'weight')
        append_meta(margin_parts, 'base_margin')
        append_meta(qid_parts, 'qid')
        append_meta(ll_parts, 'label_lower_bound')
        append_meta(lu_parts, 'label_upper_bound')
        packed_parts: List[Dict[str, Delayed]] = []
        for i in range(len(X_parts)):
            part_dict: Dict[str, Delayed] = {}
            for key, value in parts.items():
                part_dict[key] = value[i]
            packed_parts.append(part_dict)
        delayed_parts: List[Delayed] = list(map(dask.delayed, packed_parts))
        fut_parts: List[distributed.Future] = client.compute(delayed_parts)
        await distributed.wait(fut_parts)
        for part in fut_parts:
            assert part.status == 'finished', part.status
        self.partition_order = {}
        for i, part in enumerate(fut_parts):
            self.partition_order[part.key] = i
        key_to_partition = {part.key: part for part in fut_parts}
        who_has: Dict[str, Tuple[str, ...]] = await client.scheduler.who_has(keys=[part.key for part in fut_parts])
        worker_map: Dict[str, List[distributed.Future]] = defaultdict(list)
        for key, workers in who_has.items():
            worker_map[next(iter(workers))].append(key_to_partition[key])
        self.worker_map = worker_map
        if feature_weights is None:
            self.feature_weights = None
        else:
            self.feature_weights = await client.compute(feature_weights).result()
        return self

    def _create_fn_args(self, worker_addr: str) -> Dict[str, Any]:
        """Create a dictionary of objects that can be pickled for function
        arguments.

        """
        return {'feature_names': self.feature_names, 'feature_types': self.feature_types, 'feature_weights': self.feature_weights, 'missing': self.missing, 'enable_categorical': self.enable_categorical, 'parts': self.worker_map.get(worker_addr, None), 'is_quantile': self.is_quantile}

    def num_col(self) -> int:
        """Get the number of columns (features) in the DMatrix.

        Returns
        -------
        number of columns
        """
        return self._n_cols