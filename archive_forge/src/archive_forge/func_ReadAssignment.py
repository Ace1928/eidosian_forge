from sys import version_info as _swig_python_version_info
import weakref
def ReadAssignment(self, file_name):
    """
        Reads an assignment from a file and returns the current solution.
        Returns nullptr if the file cannot be opened or if the assignment is not
        valid.
        """
    return _pywrapcp.RoutingModel_ReadAssignment(self, file_name)