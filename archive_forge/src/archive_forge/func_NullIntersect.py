from sys import version_info as _swig_python_version_info
import weakref
def NullIntersect(self, first_vars, second_vars):
    """
        Creates a constraint that states that all variables in the first
        vector are different from all variables in the second
        group. Thus the set of values in the first vector does not
        intersect with the set of values in the second vector.
        """
    return _pywrapcp.Solver_NullIntersect(self, first_vars, second_vars)