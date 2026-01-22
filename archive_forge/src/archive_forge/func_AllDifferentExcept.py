from sys import version_info as _swig_python_version_info
import weakref
def AllDifferentExcept(self, vars, escape_value):
    """
        All variables are pairwise different, unless they are assigned to
        the escape value.
        """
    return _pywrapcp.Solver_AllDifferentExcept(self, vars, escape_value)