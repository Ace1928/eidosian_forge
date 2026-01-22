import inspect
import textwrap
class MouseTrap(PyomoException, NotImplementedError):
    """
    Exception class used to throw errors for not-implemented functionality
    that might be rational to support (i.e., we already gave you a cookie)
    but risks taking Pyomo's flexibility a step beyond what is sane,
    or solvable, or communicable to a solver, etc. (i.e., Really? Now you
    want a glass of milk too?)
    """

    def __str__(self):
        return format_exception(repr(super().__str__()), prolog='Sorry, mouse, no cookies here!', epilog='This is functionality we think may be rational to support, but is not yet implemented (possibly due to developer availability, complexity of edge cases, or general practicality or tractability). However, please feed the mice: pull requests are always welcome!', exception=self)