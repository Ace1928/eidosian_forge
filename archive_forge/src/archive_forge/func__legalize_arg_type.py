from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
def _legalize_arg_type(self, name, typ, loc):
    """Legalize the argument type

        Parameters
        ----------
        name: str
            argument name.
        typ: numba.core.types.Type
            argument type.
        loc: numba.core.ir.Loc
            source location for error reporting.
        """
    if getattr(typ, 'reflected', False):
        msgbuf = ['Objmode context failed.', f'Argument {name!r} is declared as an unsupported type: {typ}.', f'Reflected types are not supported.']
        raise errors.CompilerError(' '.join(msgbuf), loc=loc)