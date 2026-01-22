from sys import version_info as _swig_python_version_info
import weakref
def Accept(self, monitor, delta, deltadelta, objective_min, objective_max):
    """
        Returns true iff all filters return true, and the sum of their accepted
        objectives is between objective_min and objective_max.
        The monitor has its Begin/EndFiltering events triggered.
        """
    return _pywrapcp.LocalSearchFilterManager_Accept(self, monitor, delta, deltadelta, objective_min, objective_max)