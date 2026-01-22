import operator
import warnings
import numpy
import cupy
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import runtime
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import SparseEfficiencyWarning
from cupyx.scipy.sparse import _util
@cupy._util.memoize(for_each_device=True)
def _cupy_csr_diagonal():
    return cupy.ElementwiseKernel('int32 k, int32 rows, int32 cols, raw T data, raw I indptr, raw I indices', 'T y', '\n        int row = i;\n        int col = i;\n        if (k < 0) row -= k;\n        if (k > 0) col += k;\n        if (row >= rows || col >= cols) return;\n        int j = find_index_holding_col_in_row(row, col,\n            &(indptr[0]), &(indices[0]));\n        if (j >= 0) {\n            y = data[j];\n        } else {\n            y = static_cast<T>(0);\n        }\n        ', 'cupyx_scipy_sparse_csr_diagonal', preamble=_FIND_INDEX_HOLDING_COL_IN_ROW_)