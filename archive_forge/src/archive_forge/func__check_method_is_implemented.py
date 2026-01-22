from xarray import apply_ufunc
from ..stats import wrap_xarray_ufunc as _wrap_xarray_ufunc
def _check_method_is_implemented(self, method, *args):
    """Check a given method is implemented."""
    try:
        getattr(self, method)(*args)
    except NotImplementedError:
        return False
    except:
        return True
    return True