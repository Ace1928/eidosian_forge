from sys import version_info as _swig_python_version_info
import weakref
def AddWeightedVariableMinimizedByFinalizer(self, var, cost):
    """
        Adds a variable to minimize in the solution finalizer, with a weighted
        priority: the higher the more priority it has.
        """
    return _pywrapcp.RoutingModel_AddWeightedVariableMinimizedByFinalizer(self, var, cost)