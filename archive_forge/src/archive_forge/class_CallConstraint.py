import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
class CallConstraint(object):
    """Constraint for calling functions.
    Perform case analysis foreach combinations of argument types.
    """
    signature = None

    def __init__(self, target, func, args, kws, vararg, loc):
        self.target = target
        self.func = func
        self.args = args
        self.kws = kws or {}
        self.vararg = vararg
        self.loc = loc

    def __call__(self, typeinfer):
        msg = 'typing of call at {0}\n'.format(self.loc)
        with new_error_context(msg):
            typevars = typeinfer.typevars
            with new_error_context('resolving caller type: {}'.format(self.func)):
                fnty = typevars[self.func].getone()
            with new_error_context('resolving callee type: {0}', fnty):
                self.resolve(typeinfer, typevars, fnty)

    def resolve(self, typeinfer, typevars, fnty):
        assert fnty
        context = typeinfer.context
        r = fold_arg_vars(typevars, self.args, self.vararg, self.kws)
        if r is None:
            return
        pos_args, kw_args = r
        for a in itertools.chain(pos_args, kw_args.values()):
            if not a.is_precise() and (not isinstance(a, types.Array)):
                return
        if isinstance(fnty, types.TypeRef):
            fnty = fnty.instance_type
        try:
            sig = typeinfer.resolve_call(fnty, pos_args, kw_args)
        except ForceLiteralArg as e:
            folding_args = (fnty.this,) + tuple(self.args) if isinstance(fnty, types.BoundFunction) else self.args
            folded = e.fold_arguments(folding_args, self.kws)
            requested = set()
            unsatisfied = set()
            for idx in e.requested_args:
                maybe_arg = typeinfer.func_ir.get_definition(folded[idx])
                if isinstance(maybe_arg, ir.Arg):
                    requested.add(maybe_arg.index)
                else:
                    unsatisfied.add(idx)
            if unsatisfied:
                raise TypingError('Cannot request literal type.', loc=self.loc)
            elif requested:
                raise ForceLiteralArg(requested, loc=self.loc)
        if sig is None:
            headtemp = 'Invalid use of {0} with parameters ({1})'
            args = [str(a) for a in pos_args]
            args += ['%s=%s' % (k, v) for k, v in sorted(kw_args.items())]
            head = headtemp.format(fnty, ', '.join(map(str, args)))
            desc = context.explain_function_type(fnty)
            msg = '\n'.join([head, desc])
            raise TypingError(msg)
        typeinfer.add_type(self.target, sig.return_type, loc=self.loc)
        if isinstance(fnty, types.BoundFunction) and sig.recvr is not None and (sig.recvr != fnty.this):
            refined_this = context.unify_pairs(sig.recvr, fnty.this)
            if refined_this is None and fnty.this.is_precise() and sig.recvr.is_precise():
                msg = 'Cannot refine type {} to {}'.format(sig.recvr, fnty.this)
                raise TypingError(msg, loc=self.loc)
            if refined_this is not None and refined_this.is_precise():
                refined_fnty = fnty.copy(this=refined_this)
                typeinfer.propagate_refined_type(self.func, refined_fnty)
        if not sig.return_type.is_precise():
            target = typevars[self.target]
            if target.defined:
                targetty = target.getone()
                if context.unify_pairs(targetty, sig.return_type) == targetty:
                    sig = sig.replace(return_type=targetty)
        self.signature = sig
        self._add_refine_map(typeinfer, typevars, sig)

    def _add_refine_map(self, typeinfer, typevars, sig):
        """Add this expression to the refine_map base on the type of target_type
        """
        target_type = typevars[self.target].getone()
        if isinstance(target_type, types.Array) and isinstance(sig.return_type.dtype, types.Undefined):
            typeinfer.refine_map[self.target] = self
        if isinstance(target_type, types.DictType) and (not target_type.is_precise()):
            typeinfer.refine_map[self.target] = self

    def refine(self, typeinfer, updated_type):
        if self.func == operator.getitem:
            aryty = typeinfer.typevars[self.args[0].name].getone()
            if _is_array_not_precise(aryty):
                assert updated_type.is_precise()
                newtype = aryty.copy(dtype=updated_type.dtype)
                typeinfer.add_type(self.args[0].name, newtype, loc=self.loc)
        else:
            m = 'no type refinement implemented for function {} updating to {}'
            raise TypingError(m.format(self.func, updated_type))

    def get_call_signature(self):
        return self.signature