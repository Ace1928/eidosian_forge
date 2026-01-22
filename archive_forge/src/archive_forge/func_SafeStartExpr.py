from sys import version_info as _swig_python_version_info
import weakref
def SafeStartExpr(self, unperformed_value):
    """
        These methods create expressions encapsulating the start, end
        and duration of the interval var. If the interval var is
        unperformed, they will return the unperformed_value.
        """
    return _pywrapcp.IntervalVar_SafeStartExpr(self, unperformed_value)