import inspect
import textwrap
class DeveloperError(PyomoException, NotImplementedError):
    """
    Exception class used to throw errors that result from Pyomo
    programming errors, rather than user modeling errors (e.g., a
    component not declaring a 'ctype').
    """

    def __str__(self):
        return format_exception(repr(super().__str__()), prolog='Internal Pyomo implementation error:', epilog='Please report this to the Pyomo Developers.', exception=self)