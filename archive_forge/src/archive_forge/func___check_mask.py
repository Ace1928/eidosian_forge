import numbers
from . import _cluster  # type: ignore
def __check_mask(mask, shape):
    if mask is None:
        return np.ones(shape, dtype='intc')
    elif isinstance(mask, np.ndarray):
        return np.require(mask, dtype='intc', requirements='C')
    else:
        return np.array(mask, dtype='intc')