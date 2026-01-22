from sys import version_info as _swig_python_version_info
import weakref
def SetTransitionTime(self, transition_time):
    """
        Add a transition time between intervals.  It forces the distance between
        the end of interval a and start of interval b that follows it to be at
        least transition_time(a, b). This function must always return
        a positive or null value.
        """
    return _pywrapcp.DisjunctiveConstraint_SetTransitionTime(self, transition_time)