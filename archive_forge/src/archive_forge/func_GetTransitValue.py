from sys import version_info as _swig_python_version_info
import weakref
def GetTransitValue(self, from_index, to_index, vehicle):
    """
        Returns the transition value for a given pair of nodes (as var index);
        this value is the one taken by the corresponding transit variable when
        the 'next' variable for 'from_index' is bound to 'to_index'.
        """
    return _pywrapcp.RoutingDimension_GetTransitValue(self, from_index, to_index, vehicle)