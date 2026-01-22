from sys import version_info as _swig_python_version_info
import weakref
def SearchLeftDepth(self):
    """
        Gets the search left depth of the current active search. Returns -1 if
        there is no active search opened.
        """
    return _pywrapcp.Solver_SearchLeftDepth(self)