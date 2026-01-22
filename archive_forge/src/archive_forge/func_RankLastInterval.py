from sys import version_info as _swig_python_version_info
import weakref
def RankLastInterval(self, sequence, index):
    """
        Returns a decision that tries to rank last the ith interval var
        in the sequence variable.
        """
    return _pywrapcp.Solver_RankLastInterval(self, sequence, index)