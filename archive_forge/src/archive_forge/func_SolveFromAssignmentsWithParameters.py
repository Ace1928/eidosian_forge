from sys import version_info as _swig_python_version_info
import weakref
def SolveFromAssignmentsWithParameters(self, assignments, search_parameters, solutions=None):
    """
        Same as above but will try all assignments in order as first solutions
        until one succeeds.
        """
    return _pywrapcp.RoutingModel_SolveFromAssignmentsWithParameters(self, assignments, search_parameters, solutions)