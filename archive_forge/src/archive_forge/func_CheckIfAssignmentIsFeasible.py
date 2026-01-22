from sys import version_info as _swig_python_version_info
import weakref
def CheckIfAssignmentIsFeasible(self, assignment, call_at_solution_monitors):
    """
        Returns a vector cumul_bounds, for which cumul_bounds[i][j] is a pair
        containing the minimum and maximum of the CumulVar of the jth node on
        route i.
        - cumul_bounds[i][j].first is the minimum.
        - cumul_bounds[i][j].second is the maximum.
        Checks if an assignment is feasible.
        """
    return _pywrapcp.RoutingModel_CheckIfAssignmentIsFeasible(self, assignment, call_at_solution_monitors)