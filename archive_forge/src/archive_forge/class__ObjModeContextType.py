from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
class _ObjModeContextType(WithContext):
    """Creates a contextmanager to be used inside jitted functions to enter
    *object-mode* for using interpreter features.  The body of the with-context
    is lifted into a function that is compiled in *object-mode*.  This
    transformation process is limited and cannot process all possible
    Python code.  However, users can wrap complicated logic in another
    Python function, which will then be executed by the interpreter.

    Use this as a function that takes keyword arguments only.
    The argument names must correspond to the output variables from the
    with-block.  Their respective values can be:

    1. strings representing the expected types; i.e. ``"float32"``.
    2. compile-time bound global or nonlocal variables referring to the
       expected type. The variables are read at compile time.

    When exiting the with-context, the output variables are converted
    to the expected nopython types according to the annotation.  This process
    is the same as passing Python objects into arguments of a nopython
    function.

    Example::

        import numpy as np
        from numba import njit, objmode, types

        def bar(x):
            # This code is executed by the interpreter.
            return np.asarray(list(reversed(x.tolist())))

        # Output type as global variable
        out_ty = types.intp[:]

        @njit
        def foo():
            x = np.arange(5)
            y = np.zeros_like(x)
            with objmode(y='intp[:]', z=out_ty):  # annotate return type
                # this region is executed by object-mode.
                y += bar(x)
                z = y
            return y, z

    .. note:: Known limitations:

        - with-block cannot use incoming list objects.
        - with-block cannot use incoming function objects.
        - with-block cannot ``yield``, ``break``, ``return`` or ``raise``           such that the execution will leave the with-block immediately.
        - with-block cannot contain `with` statements.
        - random number generator states do not synchronize; i.e.           nopython-mode and object-mode uses different RNG states.

    .. note:: When used outside of no-python mode, the context-manager has no
        effect.

    .. warning:: This feature is experimental.  The supported features may
        change with or without notice.

    """
    is_callable = True

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

    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra):
        cellnames = func_ir.func_id.func.__code__.co_freevars
        closures = func_ir.func_id.func.__closure__
        func_globals = func_ir.func_id.func.__globals__
        if closures is not None:
            func_closures = {}
            for cellname, closure in zip(cellnames, closures):
                try:
                    cellval = closure.cell_contents
                except ValueError as e:
                    if str(e) != 'Cell is empty':
                        raise
                else:
                    func_closures[cellname] = cellval
        else:
            func_closures = {}
        args = extra['args'] if extra else ()
        kwargs = extra['kwargs'] if extra else {}
        typeanns = self._legalize_args(func_ir=func_ir, args=args, kwargs=kwargs, loc=blocks[blk_start].loc, func_globals=func_globals, func_closures=func_closures)
        vlt = func_ir.variable_lifetime
        inputs, outputs = find_region_inout_vars(blocks=blocks, livemap=vlt.livemap, callfrom=blk_start, returnto=blk_end, body_block_ids=set(body_blocks))

        def strip_var_ver(x):
            return x.split('.', 1)[0]
        stripped_outs = list(map(strip_var_ver, outputs))
        extra_annotated = set(typeanns) - set(stripped_outs)
        if extra_annotated:
            msg = 'Invalid type annotation on non-outgoing variables: {}.Suggestion: remove annotation of the listed variables'
            raise errors.TypingError(msg.format(extra_annotated))
        typeanns['$cp'] = types.int32
        not_annotated = set(stripped_outs) - set(typeanns)
        if not_annotated:
            msg = "Missing type annotation on outgoing variable(s): {0}\n\nExample code: with objmode({1}='<add_type_as_string_here>')\n"
            stable_ann = sorted(not_annotated)
            raise errors.TypingError(msg.format(stable_ann, stable_ann[0]))
        outtup = types.Tuple([typeanns[v] for v in stripped_outs])
        lifted_blks = {k: blocks[k] for k in body_blocks}
        _mutate_with_block_callee(lifted_blks, blk_start, blk_end, inputs, outputs)
        lifted_ir = func_ir.derive(blocks=lifted_blks, arg_names=tuple(inputs), arg_count=len(inputs), force_non_generator=True)
        dispatcher = dispatcher_factory(lifted_ir, objectmode=True, output_types=outtup)
        newblk = _mutate_with_block_caller(dispatcher, blocks, blk_start, blk_end, inputs, outputs)
        blocks[blk_start] = newblk
        _clear_blocks(blocks, body_blocks)
        return dispatcher

    def __call__(self, *args, **kwargs):
        return self