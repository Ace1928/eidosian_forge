from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _can_print(o):
    """Return True if type o can be printed with one of the SymPy printers.

        If o is a container type, this is True if and only if every element of
        o can be printed in this way.
        """
    try:
        builtin_types = (list, tuple, set, frozenset)
        if isinstance(o, builtin_types):
            if type(o).__str__ not in (i.__str__ for i in builtin_types) or type(o).__repr__ not in (i.__repr__ for i in builtin_types):
                return False
            return all((_can_print(i) for i in o))
        elif isinstance(o, dict):
            return all((_can_print(i) and _can_print(o[i]) for i in o))
        elif isinstance(o, bool):
            return False
        elif isinstance(o, Printable):
            return True
        elif any((hasattr(o, hook) for hook in printing_hooks)):
            return True
        elif isinstance(o, (float, int)) and print_builtin:
            return True
        return False
    except RuntimeError:
        return False