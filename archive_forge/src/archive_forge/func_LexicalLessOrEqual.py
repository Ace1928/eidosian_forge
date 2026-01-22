from sys import version_info as _swig_python_version_info
import weakref
def LexicalLessOrEqual(self, left, right):
    """
        Creates a constraint that enforces that left is lexicographically less
        than or equal to right.
        """
    return _pywrapcp.Solver_LexicalLessOrEqual(self, left, right)