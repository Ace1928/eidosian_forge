from sys import version_info as _swig_python_version_info
import weakref
def LocalOptimum(self):
    """
        When a local optimum is reached. If 'true' is returned, the last solution
        is discarded and the search proceeds with the next one.
        """
    return _pywrapcp.SearchMonitor_LocalOptimum(self)