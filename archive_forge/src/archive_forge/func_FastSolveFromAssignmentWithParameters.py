from sys import version_info as _swig_python_version_info
import weakref
def FastSolveFromAssignmentWithParameters(self, assignment, search_parameters, check_solution_in_cp, touched=None):
    """
        Improves a given assignment using unchecked local search.
        If check_solution_in_cp is true the final solution will be checked with
        the CP solver.
        As of 11/2023, only works with greedy descent.
        """
    return _pywrapcp.RoutingModel_FastSolveFromAssignmentWithParameters(self, assignment, search_parameters, check_solution_in_cp, touched)