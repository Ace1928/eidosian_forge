import autoray as ar
from .is_independent import is_independent
from .matrix_manipulation import expand_matrix, reduce_matrices, get_batch_size
from .multi_dispatch import (
from .quantum import (
from .fidelity import fidelity, fidelity_statevector
from .utils import (
class NumpyMimic(ar.autoray.NumpyMimic):
    """Subclass of the Autoray NumpyMimic class in order to support
    the NumPy fft submodule"""

    def __getattribute__(self, fn):
        if fn == 'fft':
            return numpy_fft
        return super().__getattribute__(fn)