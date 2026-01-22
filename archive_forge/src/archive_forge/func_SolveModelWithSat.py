from sys import version_info as _swig_python_version_info
import weakref
def SolveModelWithSat(model, search_parameters, initial_solution, solution):
    """
    Attempts to solve the model using the cp-sat solver. As of 5/2019, will
    solve the TSP corresponding to the model if it has a single vehicle.
    Therefore the resulting solution might not actually be feasible. Will return
    false if a solution could not be found.
    """
    return _pywrapcp.SolveModelWithSat(model, search_parameters, initial_solution, solution)