from sys import version_info as _swig_python_version_info
import weakref
def Synchronize(self, assignment, delta):
    """
        This method should not be overridden. Override OnSynchronize() instead
        which is called before exiting this method.
        """
    return _pywrapcp.IntVarLocalSearchFilter_Synchronize(self, assignment, delta)