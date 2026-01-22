from sys import version_info as _swig_python_version_info
import weakref
def GetFixedCostOfVehicle(self, vehicle):
    """
        Returns the route fixed cost taken into account if the route of the
        vehicle is not empty, aka there's at least one node on the route other
        than the first and last nodes.
        """
    return _pywrapcp.RoutingModel_GetFixedCostOfVehicle(self, vehicle)