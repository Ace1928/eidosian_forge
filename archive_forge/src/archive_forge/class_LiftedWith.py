import collections
import functools
import sys
import types as pytypes
import uuid
import weakref
from contextlib import ExitStack
from abc import abstractmethod
from numba import _dispatcher
from numba.core import (
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof
from numba.core.bytecode import get_code_object
from numba.core.caching import NullCache, FunctionCache
from numba.core import entrypoints
from numba.core.retarget import BaseRetarget
import numba.core.event as ev
class LiftedWith(LiftedCode):
    can_cache = True

    def _reduce_extras(self):
        return dict(output_types=self.output_types)

    @property
    def _numba_type_(self):
        return types.Dispatcher(self)

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This enables the resolving of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        if self._can_compile:
            self.compile(tuple(args))
        pysig = None
        func_name = self.py_func.__name__
        name = 'CallTemplate({0})'.format(func_name)
        call_template = typing.make_concrete_template(name, key=func_name, signatures=self.nopython_signatures)
        return (call_template, pysig, args, kws)

    def compile(self, sig):
        with ExitStack() as scope:
            cres = None

            def cb_compiler(dur):
                if cres is not None:
                    self._callback_add_compiler_timer(dur, cres)

            def cb_llvm(dur):
                if cres is not None:
                    self._callback_add_llvm_timer(dur, cres)
            scope.enter_context(ev.install_timer('numba:compiler_lock', cb_compiler))
            scope.enter_context(ev.install_timer('numba:llvm_lock', cb_llvm))
            scope.enter_context(global_compiler_lock)
            with self._compiling_counter:
                flags = self.flags
                args, return_type = sigutils.normalize_signature(sig)
                existing = self.overloads.get(tuple(args))
                if existing is not None:
                    return existing.entry_point
                self._pre_compile(args, return_type, flags)
                cloned_func_ir = self.func_ir.copy()
                ev_details = dict(dispatcher=self, args=args, return_type=return_type)
                with ev.trigger_event('numba:compile', data=ev_details):
                    cres = compiler.compile_ir(typingctx=self.typingctx, targetctx=self.targetctx, func_ir=cloned_func_ir, args=args, return_type=return_type, flags=flags, locals=self.locals, lifted=(), lifted_from=self.lifted_from, is_lifted_loop=True)
                    if cres.typing_error is not None and (not flags.enable_pyobject):
                        raise cres.typing_error
                    self.add_overload(cres)
                return cres.entry_point