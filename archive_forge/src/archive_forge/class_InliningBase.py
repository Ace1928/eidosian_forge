import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
class InliningBase(TestCase):
    _DEBUG = False
    inline_opt_as_bool = {'always': True, 'never': False}

    def sentinel_17_cost_model(self, func_ir):
        for blk in func_ir.blocks.values():
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.FreeVar):
                        if stmt.value.value == 17:
                            return True
        return False

    def check(self, test_impl, *args, **kwargs):
        inline_expect = kwargs.pop('inline_expect', None)
        assert inline_expect
        block_count = kwargs.pop('block_count', 1)
        assert not kwargs
        for k, v in inline_expect.items():
            assert isinstance(k, str)
            assert isinstance(v, bool)
        j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
        self.assertEqual(test_impl(*args), j_func(*args))
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        fir.blocks = ir_utils.simplify_CFG(fir.blocks)
        if self._DEBUG:
            print('FIR'.center(80, '-'))
            fir.dump()
        if block_count != 'SKIP':
            self.assertEqual(len(fir.blocks), block_count)
        block = next(iter(fir.blocks.values()))
        exprs = [x for x in block.find_exprs()]
        assert exprs
        for k, v in inline_expect.items():
            found = False
            for expr in exprs:
                if getattr(expr, 'op', False) == 'call':
                    func_defn = fir.get_definition(expr.func)
                    found |= func_defn.name == k
                elif ir_utils.is_operator_or_getitem(expr):
                    found |= expr.fn.__name__ == k
            self.assertFalse(found == v)
        return fir