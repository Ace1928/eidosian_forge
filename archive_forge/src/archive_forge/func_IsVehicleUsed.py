from sys import version_info as _swig_python_version_info
import weakref
def IsVehicleUsed(self, assignment, vehicle):
    """ Returns true if the route of 'vehicle' is non empty in 'assignment'."""
    return _pywrapcp.RoutingModel_IsVehicleUsed(self, assignment, vehicle)