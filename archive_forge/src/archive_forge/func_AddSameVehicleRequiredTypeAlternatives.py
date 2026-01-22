from sys import version_info as _swig_python_version_info
import weakref
def AddSameVehicleRequiredTypeAlternatives(self, dependent_type, required_type_alternatives):
    """
        Requirements:
        NOTE: As of 2019-04, cycles in the requirement graph are not supported,
        and lead to the dependent nodes being skipped if possible (otherwise
        the model is considered infeasible).
        The following functions specify that "dependent_type" requires at least
        one of the types in "required_type_alternatives".

        For same-vehicle requirements, a node of dependent type type_D requires at
        least one node of type type_R among the required alternatives on the same
        route.
        """
    return _pywrapcp.RoutingModel_AddSameVehicleRequiredTypeAlternatives(self, dependent_type, required_type_alternatives)