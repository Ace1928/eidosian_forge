from sys import version_info as _swig_python_version_info
import weakref
def SetFixedCostOfAllVehicles(self, cost):
    """
        Sets the fixed cost of all vehicle routes. It is equivalent to calling
        SetFixedCostOfVehicle on all vehicle routes.
        """
    return _pywrapcp.RoutingModel_SetFixedCostOfAllVehicles(self, cost)