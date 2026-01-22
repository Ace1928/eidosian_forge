from sys import version_info as _swig_python_version_info
import weakref
def ConstraintAdder(self, ct):
    """
        Returns a decision builder that will add the given constraint to
        the model.
        """
    return _pywrapcp.Solver_ConstraintAdder(self, ct)