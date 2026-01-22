from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _to_numpy_jax(x):
    from jax.errors import TracerArrayConversionError
    try:
        return np.array(getattr(x, 'val', x))
    except TracerArrayConversionError as e:
        raise ValueError('Converting a JAX array to a NumPy array not supported when using the JAX JIT.') from e