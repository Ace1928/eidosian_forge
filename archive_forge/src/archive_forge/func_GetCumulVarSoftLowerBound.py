from sys import version_info as _swig_python_version_info
import weakref
def GetCumulVarSoftLowerBound(self, index):
    """
        Returns the soft lower bound of a cumul variable for a given variable
        index. The "hard" lower bound of the variable is returned if no soft lower
        bound has been set.
        """
    return _pywrapcp.RoutingDimension_GetCumulVarSoftLowerBound(self, index)