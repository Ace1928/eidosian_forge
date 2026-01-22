from sys import version_info as _swig_python_version_info
import weakref
def GetCumulVarSoftUpperBoundCoefficient(self, index):
    """
        Returns the cost coefficient of the soft upper bound of a cumul variable
        for a given variable index. If no soft upper bound has been set, 0 is
        returned.
        """
    return _pywrapcp.RoutingDimension_GetCumulVarSoftUpperBoundCoefficient(self, index)