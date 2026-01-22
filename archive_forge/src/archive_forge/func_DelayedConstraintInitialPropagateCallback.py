from sys import version_info as _swig_python_version_info
import weakref
def DelayedConstraintInitialPropagateCallback(self, ct):
    """
        This method is a specialized case of the MakeConstraintDemon
        method to call the InitiatePropagate of the constraint 'ct' with
        low priority.
        """
    return _pywrapcp.Solver_DelayedConstraintInitialPropagateCallback(self, ct)