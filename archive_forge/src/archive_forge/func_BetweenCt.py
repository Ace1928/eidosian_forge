from sys import version_info as _swig_python_version_info
import weakref
def BetweenCt(self, expr, l, u):
    """ (l <= expr <= u)"""
    return _pywrapcp.Solver_BetweenCt(self, expr, l, u)