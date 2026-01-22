import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4
import cloudpickle
import numpy as np
import pandas as pd
from fugue import (
from triad import FileSystem, ParamDict, assert_or_throw, to_uuid
from tune._utils import from_base64, to_base64
from tune.concepts.flow import Trial
from tune.concepts.space import Space
from tune.constants import (
from tune.exceptions import TuneCompileError
class TuneDataset:
    """A Fugue :class:`~fugue.workflow.workflow.WorkflowDataFrame` with metadata
    representing all dataframes required for a tuning task.

    :param data: the Fugue :class:`~fugue.workflow.workflow.WorkflowDataFrame`
      containing all required dataframes
    :param dfs: the names of the dataframes
    :param keys: the common partition keys of all dataframes

    .. attention::

        Do not construct this class directly, please read
        :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`
        to find the right way
    """

    def __init__(self, data: WorkflowDataFrame, dfs: List[str], keys: List[str]):
        self._data = data.persist()
        self._dfs = dfs
        self._keys = keys

    @property
    def data(self) -> WorkflowDataFrame:
        """the Fugue :class:`~fugue.workflow.workflow.WorkflowDataFrame`
        containing all required dataframes
        """
        return self._data

    @property
    def dfs(self) -> List[str]:
        """All dataframe names (you can also find them part of the
        column names of :meth:`.data` )
        """
        return self._dfs

    @property
    def keys(self) -> List[str]:
        """Partition keys (columns) of :meth:`.data`"""
        return self._keys

    def split(self, weights: List[float], seed: Any) -> List['TuneDataset']:
        """Split the dataset randomly to small partitions. This is useful for
        some algorithms such as Hyperband, because it needs different subset to
        run successive halvings with different parameters.

        :param weights: a list of numeric values. The length represents the number
          of splitd partitions, and the values represents the proportion of each
          partition
        :param seed: random seed for the split

        :returns: a list of sub-datasets

        .. code-block:: python

            # randomly split the data to two partitions 25% and 75%
            dataset.split([1, 3], seed=0)
            # same because weights will be normalized
            dataset.split([10, 30], seed=0)

        """

        def label(df: pd.DataFrame) -> pd.DataFrame:
            if seed is not None:
                np.random.seed(seed)
            w = np.array(weights)
            p = w / np.sum(w)
            df['__tune_split_id_'] = np.random.choice(len(weights), df.shape[0], p=p)
            return df.reset_index(drop=True)

        def select(df: pd.DataFrame, n: int) -> pd.DataFrame:
            return df[df['__tune_split_id_'] == n].drop(['__tune_split_id_'], axis=1).reset_index(drop=True)
        temp = self._data.process(label).persist()
        datasets: List['TuneDataset'] = []
        for i in range(len(weights)):
            datasets.append(TuneDataset(temp.transform(select, schema='*-__tune_split_id_', params=dict(n=i)), self.dfs, self.keys))
        return datasets