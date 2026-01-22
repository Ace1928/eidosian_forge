from sys import version_info as _swig_python_version_info
import weakref
def IndexExpression(self, vars, value):
    """
        Returns the expression expr such that vars[expr] == value.
        It assumes that vars are all different.
        """
    return _pywrapcp.Solver_IndexExpression(self, vars, value)