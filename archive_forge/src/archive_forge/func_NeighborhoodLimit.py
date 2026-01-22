from sys import version_info as _swig_python_version_info
import weakref
def NeighborhoodLimit(self, op, limit):
    """
        Creates a local search operator that wraps another local search
        operator and limits the number of neighbors explored (i.e., calls
        to MakeNextNeighbor from the current solution (between two calls
        to Start()). When this limit is reached, MakeNextNeighbor()
        returns false. The counter is cleared when Start() is called.
        """
    return _pywrapcp.Solver_NeighborhoodLimit(self, op, limit)