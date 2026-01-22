from sys import version_info as _swig_python_version_info
import weakref
def SetArcCostEvaluatorOfVehicle(self, evaluator_index, vehicle):
    """ Sets the cost function for a given vehicle route."""
    return _pywrapcp.RoutingModel_SetArcCostEvaluatorOfVehicle(self, evaluator_index, vehicle)