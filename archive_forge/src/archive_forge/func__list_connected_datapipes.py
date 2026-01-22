import io
import pickle
import warnings
from collections.abc import Collection
from typing import Dict, List, Optional, Set, Tuple, Type, Union
from torch.utils.data import IterDataPipe, MapDataPipe
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def _list_connected_datapipes(scan_obj: DataPipe, only_datapipe: bool, cache: Set[int]) -> List[DataPipe]:
    f = io.BytesIO()
    p = pickle.Pickler(f)
    if DILL_AVAILABLE:
        from dill import Pickler as dill_Pickler
        d = dill_Pickler(f)
    else:
        d = None
    captured_connections = []

    def getstate_hook(ori_state):
        state = None
        if isinstance(ori_state, dict):
            state = {}
            for k, v in ori_state.items():
                if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                    state[k] = v
        elif isinstance(ori_state, (tuple, list)):
            state = []
            for v in ori_state:
                if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                    state.append(v)
        elif isinstance(ori_state, (IterDataPipe, MapDataPipe, Collection)):
            state = ori_state
        return state

    def reduce_hook(obj):
        if obj == scan_obj or id(obj) in cache:
            raise NotImplementedError
        else:
            captured_connections.append(obj)
            cache.add(id(obj))
            return (_stub_unpickler, ())
    datapipe_classes: Tuple[Type[DataPipe]] = (IterDataPipe, MapDataPipe)
    try:
        for cls in datapipe_classes:
            cls.set_reduce_ex_hook(reduce_hook)
            if only_datapipe:
                cls.set_getstate_hook(getstate_hook)
        try:
            p.dump(scan_obj)
        except (pickle.PickleError, AttributeError, TypeError):
            if DILL_AVAILABLE:
                d.dump(scan_obj)
            else:
                raise
    finally:
        for cls in datapipe_classes:
            cls.set_reduce_ex_hook(None)
            if only_datapipe:
                cls.set_getstate_hook(None)
        if DILL_AVAILABLE:
            from dill import extend as dill_extend
            dill_extend(False)
    return captured_connections