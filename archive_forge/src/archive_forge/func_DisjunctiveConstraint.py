from sys import version_info as _swig_python_version_info
import weakref
def DisjunctiveConstraint(self, intervals, name):
    """
        This constraint forces all interval vars into an non-overlapping
        sequence. Intervals with zero duration can be scheduled anywhere.
        """
    return _pywrapcp.Solver_DisjunctiveConstraint(self, intervals, name)