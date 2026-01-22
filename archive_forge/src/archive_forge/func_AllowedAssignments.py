from sys import version_info as _swig_python_version_info
import weakref
def AllowedAssignments(self, *args):
    """
        *Overload 1:*
        This method creates a constraint where the graph of the relation
        between the variables is given in extension. There are 'arity'
        variables involved in the relation and the graph is given by a
        integer tuple set.

        |

        *Overload 2:*
        Compatibility layer for Python API.
        """
    return _pywrapcp.Solver_AllowedAssignments(self, *args)