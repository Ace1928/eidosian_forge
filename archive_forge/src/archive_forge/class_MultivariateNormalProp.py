from distutils.version import StrictVersion
import functools
import ast
import numpy as np
from . import operator
from . import numpy as _mx_np  # pylint: disable=reimported
from .util import np_array, use_np
from .numpy.utils import _STR_2_DTYPE_
from .ndarray.numpy import _internal as _nd_npi
from .symbol.numpy import _internal as _sym_npi
@register('mvn_fallback')
class MultivariateNormalProp(operator.CustomOpProp):
    """Fallback np.random.multivariate_normal operator properties."""

    def __init__(self, size=None):
        super(MultivariateNormalProp, self).__init__(need_top_grad=True)
        self._size = ast.literal_eval(size) if size is not None else None

    def list_arguments(self):
        return ['mean', 'cov']

    def infer_shape(self, in_shape):
        loc_shape = in_shape[0]
        cov_shape = in_shape[1]
        if len(loc_shape) < 1:
            raise ValueError('mean must be at least 1 dimensional')
        if len(cov_shape) < 2:
            raise ValueError('cov must be at least 2 dimensional')
        if cov_shape[-1] != cov_shape[-2]:
            raise ValueError('the last two dimentions of the parameter cov have to be the same, whereas the shape of cov is {}'.format(cov_shape))
        if cov_shape[-1] != loc_shape[-1]:
            raise ValueError('mean and cov must have same length.The shape of mean is {} but the shape of cov is {}'.format(loc_shape[-1:], cov_shape[-2:]))
        out_shape = np.broadcast(np.empty(loc_shape), np.empty(cov_shape[:-1])).shape
        if self._size is not None:
            self._size = (self._size,) if np.isscalar(self._size) else self._size
            out_shape = self._size + out_shape
        return (in_shape, (out_shape,), ())

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return MultivariateNormal(self._size)