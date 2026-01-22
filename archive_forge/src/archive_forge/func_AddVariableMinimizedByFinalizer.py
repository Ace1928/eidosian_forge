from sys import version_info as _swig_python_version_info
import weakref
def AddVariableMinimizedByFinalizer(self, var):
    """
        Adds a variable to minimize in the solution finalizer. The solution
        finalizer is called each time a solution is found during the search and
        allows to instantiate secondary variables (such as dimension cumul
        variables).
        """
    return _pywrapcp.RoutingModel_AddVariableMinimizedByFinalizer(self, var)