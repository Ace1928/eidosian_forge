import scipy._lib.uarray as ua
from . import _basic_backend
from . import _realtransforms_backend
from . import _fftlog_backend
def _backend_from_arg(backend):
    """Maps strings to known backends and validates the backend"""
    if isinstance(backend, str):
        try:
            backend = _named_backends[backend]
        except KeyError as e:
            raise ValueError(f'Unknown backend {backend}') from e
    if backend.__ua_domain__ != 'numpy.scipy.fft':
        raise ValueError('Backend does not implement "numpy.scipy.fft"')
    return backend