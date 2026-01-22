from sys import version_info as _swig_python_version_info
import weakref
def AddLocalSearchOperator(self, ls_operator):
    """
        Adds a local search operator to the set of operators used to solve the
        vehicle routing problem.
        """
    return _pywrapcp.RoutingModel_AddLocalSearchOperator(self, ls_operator)