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
class LiftedLoop(LiftedCode):

    def _pre_compile(self, args, return_type, flags):
        assert not flags.enable_looplift, 'Enable looplift flags is on'

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
                npm_loop_flags = flags.copy()
                npm_loop_flags.force_pyobject = False
                pyobject_loop_flags = flags.copy()
                pyobject_loop_flags.force_pyobject = True
                cloned_func_ir = self.func_ir.copy()
                ev_details = dict(dispatcher=self, args=args, return_type=return_type)
                with ev.trigger_event('numba:compile', data=ev_details):
                    try:
                        cres = compiler.compile_ir(typingctx=self.typingctx, targetctx=self.targetctx, func_ir=cloned_func_ir, args=args, return_type=return_type, flags=npm_loop_flags, locals=self.locals, lifted=(), lifted_from=self.lifted_from, is_lifted_loop=True)
                    except errors.TypingError:
                        cres = compiler.compile_ir(typingctx=self.typingctx, targetctx=self.targetctx, func_ir=cloned_func_ir, args=args, return_type=return_type, flags=pyobject_loop_flags, locals=self.locals, lifted=(), lifted_from=self.lifted_from, is_lifted_loop=True)
                    if cres.typing_error is not None:
                        raise cres.typing_error
                    self.add_overload(cres)
                return cres.entry_point