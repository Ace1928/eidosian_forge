from sys import version_info as _swig_python_version_info
import weakref
def HasTypeRegulations(self):
    """
        Returns true iff the model has any incompatibilities or requirements set
        on node types.
        """
    return _pywrapcp.RoutingModel_HasTypeRegulations(self)