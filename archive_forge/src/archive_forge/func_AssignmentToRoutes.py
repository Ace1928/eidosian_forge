from sys import version_info as _swig_python_version_info
import weakref
def AssignmentToRoutes(self, assignment, routes):
    """
        Converts the solution in the given assignment to routes for all vehicles.
        Expects that assignment contains a valid solution (i.e. routes for all
        vehicles end with an end index for that vehicle).
        """
    return _pywrapcp.RoutingModel_AssignmentToRoutes(self, assignment, routes)