from sys import version_info as _swig_python_version_info
import weakref
def CloseModelWithParameters(self, search_parameters):
    """
        Same as above taking search parameters (as of 10/2015 some the parameters
        have to be set when closing the model).
        """
    return _pywrapcp.RoutingModel_CloseModelWithParameters(self, search_parameters)