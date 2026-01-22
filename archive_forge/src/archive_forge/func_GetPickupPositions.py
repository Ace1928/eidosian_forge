from sys import version_info as _swig_python_version_info
import weakref
def GetPickupPositions(self, node_index):
    """ Returns the pickup and delivery positions where the node is a pickup."""
    return _pywrapcp.RoutingModel_GetPickupPositions(self, node_index)