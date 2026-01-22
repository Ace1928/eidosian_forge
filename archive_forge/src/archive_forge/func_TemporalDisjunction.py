from sys import version_info as _swig_python_version_info
import weakref
def TemporalDisjunction(self, *args):
    """
        *Overload 1:*
        This constraint implements a temporal disjunction between two
        interval vars t1 and t2. 'alt' indicates which alternative was
        chosen (alt == 0 is equivalent to t1 before t2).

        |

        *Overload 2:*
        This constraint implements a temporal disjunction between two
        interval vars.
        """
    return _pywrapcp.Solver_TemporalDisjunction(self, *args)