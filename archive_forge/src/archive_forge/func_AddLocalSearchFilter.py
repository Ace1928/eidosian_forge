from sys import version_info as _swig_python_version_info
import weakref
def AddLocalSearchFilter(self, filter):
    """
        Adds a custom local search filter to the list of filters used to speed up
        local search by pruning unfeasible variable assignments.
        Calling this method after the routing model has been closed (CloseModel()
        or Solve() has been called) has no effect.
        The routing model does not take ownership of the filter.
        """
    return _pywrapcp.RoutingModel_AddLocalSearchFilter(self, filter)