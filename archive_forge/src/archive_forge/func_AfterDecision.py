from sys import version_info as _swig_python_version_info
import weakref
def AfterDecision(self, d, apply):
    """
        Just after refuting or applying the decision, apply is true after Apply.
        This is called only if the Apply() or Refute() methods have not failed.
        """
    return _pywrapcp.SearchMonitor_AfterDecision(self, d, apply)