import numba
from numba.tests.support import TestCase, unittest
from numba.core.registry import cpu_target
from numba.core.compiler import CompilerBase, Flags
from numba.core.compiler_machinery import PassManager
from numba.core import types, ir, bytecode, compiler, ir_utils, registry
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference,
from numba.experimental import jitclass
def test_obj_func_match(self):
    """Test matching of an object method (other than Array see #3449)
        """

    def test_func():
        d = Dummy([1])
        d.val.append(2)
    test_ir = compiler.run_frontend(test_func)
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    typing_res = type_inference_stage(typingctx, targetctx, test_ir, (), None)
    matched_call = ir_utils.find_callname(test_ir, test_ir.blocks[0].body[7].value, typing_res.typemap)
    self.assertTrue(isinstance(matched_call, tuple) and len(matched_call) == 2 and (matched_call[0] == 'append'))