from sys import version_info as _swig_python_version_info
import weakref
def AddWeightedSumOfAssignedDimension(self, weights, cost_var):
    """
        This dimension enforces that cost_var == sum of weights[i] for
        all objects 'i' assigned to a bin.
        """
    return _pywrapcp.Pack_AddWeightedSumOfAssignedDimension(self, weights, cost_var)