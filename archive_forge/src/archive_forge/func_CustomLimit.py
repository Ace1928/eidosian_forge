from sys import version_info as _swig_python_version_info
import weakref
def CustomLimit(self, limiter):
    """
        Callback-based search limit. Search stops when limiter returns true; if
        this happens at a leaf the corresponding solution will be rejected.
        """
    return _pywrapcp.Solver_CustomLimit(self, limiter)