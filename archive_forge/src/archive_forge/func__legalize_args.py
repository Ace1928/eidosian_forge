from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
def _legalize_args(self, func_ir, args, kwargs, loc, func_globals, func_closures):
    """
        Legalize arguments to the context-manager

        Parameters
        ----------
        func_ir: FunctionIR
        args: tuple
            Positional arguments to the with-context call as IR nodes.
        kwargs: dict
            Keyword arguments to the with-context call as IR nodes.
        loc: numba.core.ir.Loc
            Source location of the with-context call.
        func_globals: dict
            The globals dictionary of the calling function.
        func_closures: dict
            The resolved closure variables of the calling function.
        """
    if args:
        raise errors.CompilerError("objectmode context doesn't take any positional arguments")
    typeanns = {}

    def report_error(varname, msg, loc):
        raise errors.CompilerError(f'Error handling objmode argument {varname!r}. {msg}', loc=loc)
    for k, v in kwargs.items():
        if isinstance(v, ir.Const) and isinstance(v.value, str):
            typeanns[k] = sigutils._parse_signature_string(v.value)
        elif isinstance(v, ir.FreeVar):
            try:
                v = func_closures[v.name]
            except KeyError:
                report_error(varname=k, msg=f'Freevar {v.name!r} is not defined.', loc=loc)
            typeanns[k] = v
        elif isinstance(v, ir.Global):
            try:
                v = func_globals[v.name]
            except KeyError:
                report_error(varname=k, msg=f'Global {v.name!r} is not defined.', loc=loc)
            typeanns[k] = v
        elif isinstance(v, ir.Expr) and v.op == 'getattr':
            try:
                base_obj = func_ir.infer_constant(v.value)
                typ = getattr(base_obj, v.attr)
            except (errors.ConstantInferenceError, AttributeError):
                report_error(varname=k, msg='Getattr cannot be resolved at compile-time.', loc=loc)
            else:
                typeanns[k] = typ
        else:
            report_error(varname=k, msg='The value must be a compile-time constant either as a non-local variable or a getattr expression that refers to a Numba type.', loc=loc)
    for name, typ in typeanns.items():
        self._legalize_arg_type(name, typ, loc)
    return typeanns