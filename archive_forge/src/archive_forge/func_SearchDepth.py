from sys import version_info as _swig_python_version_info
import weakref
def SearchDepth(self):
    """
        Gets the search depth of the current active search. Returns -1 if
        there is no active search opened.
        """
    return _pywrapcp.Solver_SearchDepth(self)