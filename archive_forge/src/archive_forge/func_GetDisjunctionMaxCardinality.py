from sys import version_info as _swig_python_version_info
import weakref
def GetDisjunctionMaxCardinality(self, index):
    """
        Returns the maximum number of possible active nodes of the node
        disjunction of index 'index'.
        """
    return _pywrapcp.RoutingModel_GetDisjunctionMaxCardinality(self, index)