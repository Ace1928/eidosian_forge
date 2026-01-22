from sys import version_info as _swig_python_version_info
import weakref
def Optimize(self, maximize, v, step):
    """ Creates a objective with a given sense (true = maximization)."""
    return _pywrapcp.Solver_Optimize(self, maximize, v, step)