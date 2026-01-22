from sys import version_info as _swig_python_version_info
import weakref
def DebugOutputAssignment(self, solution_assignment, dimension_to_print):
    """
        Print some debugging information about an assignment, including the
        feasible intervals of the CumulVar for dimension "dimension_to_print"
        at each step of the routes.
        If "dimension_to_print" is omitted, all dimensions will be printed.
        """
    return _pywrapcp.RoutingModel_DebugOutputAssignment(self, solution_assignment, dimension_to_print)