from sys import version_info as _swig_python_version_info
import weakref
def RegisterUnaryTransitVector(self, values):
    """
        Registers 'callback' and returns its index.
        The sign parameter allows to notify the solver that the callback only
        return values of the given sign. This can help the solver, but passing
        an incorrect sign may crash in non-opt compilation mode, and yield
        incorrect results in opt.
        """
    return _pywrapcp.RoutingModel_RegisterUnaryTransitVector(self, values)