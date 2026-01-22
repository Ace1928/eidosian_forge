import unittest
from contextlib import contextmanager
from llvmlite import ir
from numba.core import types, typing, callconv, cpu, cgutils
from numba.core.registry import cpu_target
@contextmanager
def _context_builder_sig_args(self):
    typing_context = cpu_target.typing_context
    context = cpu_target.target_context
    lib = context.codegen().create_library('testing')
    with context.push_code_library(lib):
        module = ir.Module('test_module')
        sig = typing.signature(types.int32, types.int32)
        llvm_fnty = context.call_conv.get_function_type(sig.return_type, sig.args)
        function = cgutils.get_or_insert_function(module, llvm_fnty, 'test_fn')
        args = context.call_conv.get_arguments(function)
        assert function.is_declaration
        entry_block = function.append_basic_block('entry')
        builder = ir.IRBuilder(entry_block)
        yield (context, builder, sig, args)