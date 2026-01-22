from sys import version_info as _swig_python_version_info
import weakref
def GetArcCostForFirstSolution(self, from_index, to_index):
    """
        Returns the cost of the arc in the context of the first solution strategy.
        This is typically a simplification of the actual cost; see the .cc.
        """
    return _pywrapcp.RoutingModel_GetArcCostForFirstSolution(self, from_index, to_index)