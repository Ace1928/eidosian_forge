from sys import version_info as _swig_python_version_info
import weakref
def LexicalLess(self, left, right):
    """
        Creates a constraint that enforces that left is lexicographically less
        than right.
        """
    return _pywrapcp.Solver_LexicalLess(self, left, right)