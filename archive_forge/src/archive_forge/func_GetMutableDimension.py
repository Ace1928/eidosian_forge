from sys import version_info as _swig_python_version_info
import weakref
def GetMutableDimension(self, dimension_name):
    """
        Returns a dimension from its name. Returns nullptr if the dimension does
        not exist.
        """
    return _pywrapcp.RoutingModel_GetMutableDimension(self, dimension_name)