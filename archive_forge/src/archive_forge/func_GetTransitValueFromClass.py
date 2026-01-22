from sys import version_info as _swig_python_version_info
import weakref
def GetTransitValueFromClass(self, from_index, to_index, vehicle_class):
    """
        Same as above but taking a vehicle class of the dimension instead of a
        vehicle (the class of a vehicle can be obtained with vehicle_to_class()).
        """
    return _pywrapcp.RoutingDimension_GetTransitValueFromClass(self, from_index, to_index, vehicle_class)