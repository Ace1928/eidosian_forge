from sys import version_info as _swig_python_version_info
import weakref
def Cover(self, vars, target_var):
    """
        This constraint states that the target_var is the convex hull of
        the intervals. If none of the interval variables is performed,
        then the target var is unperformed too. Also, if the target
        variable is unperformed, then all the intervals variables are
        unperformed too.
        """
    return _pywrapcp.Solver_Cover(self, vars, target_var)