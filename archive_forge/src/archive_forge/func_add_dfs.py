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
def add_dfs(self, dfs: WorkflowDataFrames, how: str='') -> 'TuneDatasetBuilder':
    """Add multiple dataframes with the same join type

        :param dfs: dictionary like dataframe collection. The keys
          will be used as the dataframe names
        :param how: join type, can accept ``semi``, ``left_semi``,
          ``anti``, ``left_anti``, ``inner``, ``left_outer``,
          ``right_outer``, ``full_outer``, ``cross``
        :returns: the builder itself
        """
    assert_or_throw(dfs.has_key, 'all datarames must be named')
    for k, v in dfs.items():
        if len(self._dfs_spec) == 0:
            self.add_df(k, v)
        else:
            self.add_df(k, v, how=how)
    return self