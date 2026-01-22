from sys import version_info as _swig_python_version_info
import weakref
def ConstraintInitialPropagateCallback(self, ct):
    """
        This method is a specialized case of the MakeConstraintDemon
        method to call the InitiatePropagate of the constraint 'ct'.
        """
    return _pywrapcp.Solver_ConstraintInitialPropagateCallback(self, ct)