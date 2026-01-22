from sys import version_info as _swig_python_version_info
import weakref
def VehicleVar(self, index):
    """
        Returns the vehicle variable of the node corresponding to index. Note that
        VehicleVar(index) == -1 is equivalent to ActiveVar(index) == 0.
        """
    return _pywrapcp.RoutingModel_VehicleVar(self, index)