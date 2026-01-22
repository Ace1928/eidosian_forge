from sys import version_info as _swig_python_version_info
import weakref
def SortingConstraint(self, vars, sorted):
    """
        Creates a constraint binding the arrays of variables "vars" and
        "sorted_vars": sorted_vars[0] must be equal to the minimum of all
        variables in vars, and so on: the value of sorted_vars[i] must be
        equal to the i-th value of variables invars.

        This constraint propagates in both directions: from "vars" to
        "sorted_vars" and vice-versa.

        Behind the scenes, this constraint maintains that:
          - sorted is always increasing.
          - whatever the values of vars, there exists a permutation that
            injects its values into the sorted variables.

        For more info, please have a look at:
          https://mpi-inf.mpg.de/~mehlhorn/ftp/Mehlhorn-Thiel.pdf
        """
    return _pywrapcp.Solver_SortingConstraint(self, vars, sorted)