from sys import version_info as _swig_python_version_info
import weakref
def NextVar(self, index):
    """
        Returns the next variable of the node corresponding to index. Note that
        NextVar(index) == index is equivalent to ActiveVar(index) == 0.
        """
    return _pywrapcp.RoutingModel_NextVar(self, index)