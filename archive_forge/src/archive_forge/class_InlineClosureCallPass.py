import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
class InlineClosureCallPass(object):
    """InlineClosureCallPass class looks for direct calls to locally defined
    closures, and inlines the body of the closure function to the call site.
    """

    def __init__(self, func_ir, parallel_options, swapped={}, typed=False):
        self.func_ir = func_ir
        self.parallel_options = parallel_options
        self.swapped = swapped
        self._processed_stencils = []
        self.typed = typed

    def run(self):
        """Run inline closure call pass.
        """
        pp = postproc.PostProcessor(self.func_ir)
        pp.run(True)
        modified = False
        work_list = list(self.func_ir.blocks.items())
        debug_print = _make_debug_print('InlineClosureCallPass')
        debug_print('START')
        while work_list:
            _label, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        call_name = guard(find_callname, self.func_ir, expr)
                        func_def = guard(get_definition, self.func_ir, expr.func)
                        if guard(self._inline_reduction, work_list, block, i, expr, call_name):
                            modified = True
                            break
                        if guard(self._inline_closure, work_list, block, i, func_def):
                            modified = True
                            break
                        if guard(self._inline_stencil, instr, call_name, func_def):
                            modified = True
        if enable_inline_arraycall:
            if modified:
                merge_adjacent_blocks(self.func_ir.blocks)
            cfg = compute_cfg_from_blocks(self.func_ir.blocks)
            debug_print('start inline arraycall')
            _debug_dump(cfg)
            loops = cfg.loops()
            sized_loops = [(k, len(loops[k].body)) for k in loops.keys()]
            visited = []
            for k, s in sorted(sized_loops, key=lambda tup: tup[1], reverse=True):
                visited.append(k)
                if guard(_inline_arraycall, self.func_ir, cfg, visited, loops[k], self.swapped, self.parallel_options.comprehension, self.typed):
                    modified = True
            if modified:
                _fix_nested_array(self.func_ir)
        if modified:
            cfg = compute_cfg_from_blocks(self.func_ir.blocks)
            for dead in cfg.dead_nodes():
                del self.func_ir.blocks[dead]
            dead_code_elimination(self.func_ir)
            self.func_ir.blocks = rename_labels(self.func_ir.blocks)
        remove_dels(self.func_ir.blocks)
        debug_print('END')

    def _inline_reduction(self, work_list, block, i, expr, call_name):
        require(not self.parallel_options.reduction)
        require(call_name == ('reduce', 'builtins') or call_name == ('reduce', '_functools'))
        if len(expr.args) not in (2, 3):
            raise TypeError('invalid reduce call, two arguments are required (optional initial value can also be specified)')
        check_reduce_func(self.func_ir, expr.args[0])

        def reduce_func(f, A, v=None):
            it = iter(A)
            if v is not None:
                s = v
            else:
                s = next(it)
            for a in it:
                s = f(s, a)
            return s
        inline_closure_call(self.func_ir, self.func_ir.func_id.func.__globals__, block, i, reduce_func, work_list=work_list, callee_validator=callee_ir_validator)
        return True

    def _inline_stencil(self, instr, call_name, func_def):
        from numba.stencils.stencil import StencilFunc
        lhs = instr.target
        expr = instr.value
        if isinstance(func_def, ir.Global) and func_def.name == 'stencil' and isinstance(func_def.value, StencilFunc):
            if expr.kws:
                expr.kws += func_def.value.kws
            else:
                expr.kws = func_def.value.kws
            return True
        require(call_name == ('stencil', 'numba.stencils.stencil') or call_name == ('stencil', 'numba'))
        require(expr not in self._processed_stencils)
        self._processed_stencils.append(expr)
        if not len(expr.args) == 1:
            raise ValueError('As a minimum Stencil requires a kernel as an argument')
        stencil_def = guard(get_definition, self.func_ir, expr.args[0])
        require(isinstance(stencil_def, ir.Expr) and stencil_def.op == 'make_function')
        kernel_ir = get_ir_of_code(self.func_ir.func_id.func.__globals__, stencil_def.code)
        options = dict(expr.kws)
        if 'neighborhood' in options:
            fixed = guard(self._fix_stencil_neighborhood, options)
            if not fixed:
                raise ValueError('stencil neighborhood option should be a tuple with constant structure such as ((-w, w),)')
        if 'index_offsets' in options:
            fixed = guard(self._fix_stencil_index_offsets, options)
            if not fixed:
                raise ValueError('stencil index_offsets option should be a tuple with constant structure such as (offset, )')
        sf = StencilFunc(kernel_ir, 'constant', options)
        sf.kws = expr.kws
        sf_global = ir.Global('stencil', sf, expr.loc)
        self.func_ir._definitions[lhs.name] = [sf_global]
        instr.value = sf_global
        return True

    def _fix_stencil_neighborhood(self, options):
        """
        Extract the two-level tuple representing the stencil neighborhood
        from the program IR to provide a tuple to StencilFunc.
        """
        dims_build_tuple = get_definition(self.func_ir, options['neighborhood'])
        require(hasattr(dims_build_tuple, 'items'))
        res = []
        for window_var in dims_build_tuple.items:
            win_build_tuple = get_definition(self.func_ir, window_var)
            require(hasattr(win_build_tuple, 'items'))
            res.append(tuple(win_build_tuple.items))
        options['neighborhood'] = tuple(res)
        return True

    def _fix_stencil_index_offsets(self, options):
        """
        Extract the tuple representing the stencil index offsets
        from the program IR to provide to StencilFunc.
        """
        offset_tuple = get_definition(self.func_ir, options['index_offsets'])
        require(hasattr(offset_tuple, 'items'))
        options['index_offsets'] = tuple(offset_tuple.items)
        return True

    def _inline_closure(self, work_list, block, i, func_def):
        require(isinstance(func_def, ir.Expr) and func_def.op == 'make_function')
        inline_closure_call(self.func_ir, self.func_ir.func_id.func.__globals__, block, i, func_def, work_list=work_list, callee_validator=callee_ir_validator)
        return True