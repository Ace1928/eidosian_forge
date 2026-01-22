import cupy
from cupyx import cusparse
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
import cupyx.scipy.sparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
def _convert_dense(self, x):
    if cusparse.check_availability('denseToSparse'):
        m = cusparse.denseToSparse(x, format='csc')
    else:
        m = cusparse.dense2csc(x)
    return (m.data, m.indices, m.indptr)