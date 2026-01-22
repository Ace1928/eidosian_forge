import itertools
import numpy as np
import pandas as pd
import param
from ..core import Dataset
from ..core.boundingregion import BoundingBox
from ..core.data import PandasInterface, default_datatype
from ..core.operation import Operation
from ..core.sheetcoords import Slice
from ..core.util import (
def _aggregate_dataset_pandas(self, obj):
    index_cols = [d.name for d in obj.kdims]
    df = obj.data.set_index(index_cols).groupby(index_cols, sort=False).first()
    label = 'unique' if len(df) == len(obj) else 'non-unique'
    levels = self._get_coords(obj)
    index = pd.MultiIndex.from_product(levels, names=df.index.names)
    reindexed = df.reindex(index)
    data = tuple(levels)
    shape = tuple((d.shape[0] for d in data))
    for vdim in obj.vdims:
        data += (reindexed[vdim.name].values.reshape(shape).T,)
    return obj.clone(data, datatype=self.p.datatype, label=label)