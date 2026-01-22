from collections import defaultdict
import copy
import sys
from itertools import permutations, takewhile
from contextlib import contextmanager
from functools import cached_property
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
import llvmlite.binding as ll
from numba.core import types, utils, datamodel, debuginfo, funcdesc, config, cgutils, imputils
from numba.core import event, errors, targetconfig
from numba import _dynfunc, _helperlib
from numba.core.compiler_lock import global_compiler_lock
from numba.core.pythonapi import PythonAPI
from numba.core.imputils import (user_function, user_generator,
from numba.cpython import builtins
def _compile_subroutine_no_cache(self, builder, impl, sig, locals={}, flags=None):
    """
        Invoke the compiler to compile a function to be used inside a
        nopython function, but without generating code to call that
        function.

        Note this context's flags are not inherited.
        """
    from numba.core import compiler
    with global_compiler_lock:
        codegen = self.codegen()
        library = codegen.create_library(impl.__name__)
        if flags is None:
            cstk = targetconfig.ConfigStack()
            flags = compiler.Flags()
            if cstk:
                tls_flags = cstk.top()
                if tls_flags.is_set('nrt') and tls_flags.nrt:
                    flags.nrt = True
        flags.no_compile = True
        flags.no_cpython_wrapper = True
        flags.no_cfunc_wrapper = True
        cres = compiler.compile_internal(self.typing_context, self, library, impl, sig.args, sig.return_type, flags, locals=locals)
        self.active_code_library.add_linking_library(cres.library)
        return cres