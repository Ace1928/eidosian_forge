from collections import OrderedDict
import numpy as np
from ..ndarray.sparse import CSRNDArray
from ..ndarray.sparse import array as sparse_array
from ..ndarray import NDArray
from ..ndarray import array
def _getdata_by_idx(data, idx):
    """Shuffle the data."""
    shuffle_data = []
    for k, v in data:
        if isinstance(v, h5py.Dataset) if h5py else False:
            shuffle_data.append((k, v))
        elif isinstance(v, CSRNDArray):
            shuffle_data.append((k, sparse_array(v.asscipy()[idx], v.context)))
        else:
            shuffle_data.append((k, array(v.asnumpy()[idx], v.context)))
    return shuffle_data