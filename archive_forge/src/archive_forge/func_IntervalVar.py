from sys import version_info as _swig_python_version_info
import weakref
def IntervalVar(self, start_min, start_max, duration_min, duration_max, end_min, end_max, optional, name):
    """
        Creates an interval var by specifying the bounds on start,
        duration, and end.
        """
    return _pywrapcp.Solver_IntervalVar(self, start_min, start_max, duration_min, duration_max, end_min, end_max, optional, name)