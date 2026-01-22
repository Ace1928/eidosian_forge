import inspect
import textwrap
class TempfileContextError(PyomoException, IndexError):
    """A Pyomo-specific IndexError raised when attempting to use the
    TempfileManager when it does not have a currently active context.

    """
    pass