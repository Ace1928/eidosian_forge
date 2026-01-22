from cupy import _util
from cupy.fft._cache import get_plan_cache  # NOQA
from cupy.fft._cache import clear_plan_cache  # NOQA
from cupy.fft._cache import get_plan_cache_size  # NOQA
from cupy.fft._cache import set_plan_cache_size  # NOQA
from cupy.fft._cache import get_plan_cache_max_memsize  # NOQA
from cupy.fft._cache import set_plan_cache_max_memsize  # NOQA
from cupy.fft._cache import show_plan_cache_info  # NOQA
import sys as _sys
class set_cufft_callbacks:

    def __init__(self, *args, **kwargs):
        raise RuntimeError('cuFFT callback is only available on Linux')