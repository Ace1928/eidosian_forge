from sys import version_info as _swig_python_version_info
import weakref
def FailuresLimit(self, failures):
    """
        Creates a search limit that constrains the number of failures
        that can happen when exploring the search tree.
        """
    return _pywrapcp.Solver_FailuresLimit(self, failures)