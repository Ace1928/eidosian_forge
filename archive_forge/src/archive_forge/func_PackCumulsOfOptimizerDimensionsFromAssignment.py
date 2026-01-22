from sys import version_info as _swig_python_version_info
import weakref
def PackCumulsOfOptimizerDimensionsFromAssignment(self, original_assignment, duration_limit, time_limit_was_reached=None):
    """
        For every dimension in the model with an optimizer in
        local/global_dimension_optimizers_, this method tries to pack the cumul
        values of the dimension, such that:
        - The cumul costs (span costs, soft lower and upper bound costs, etc) are
          minimized.
        - The cumuls of the ends of the routes are minimized for this given
          minimal cumul cost.
        - Given these minimal end cumuls, the route start cumuls are maximized.
        Returns the assignment resulting from allocating these packed cumuls with
        the solver, and nullptr if these cumuls could not be set by the solver.
        """
    return _pywrapcp.RoutingModel_PackCumulsOfOptimizerDimensionsFromAssignment(self, original_assignment, duration_limit, time_limit_was_reached)