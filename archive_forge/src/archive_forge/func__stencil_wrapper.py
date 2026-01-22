import copy
import numpy as np
from llvmlite import ir as lir
from numba.core import types, typing, utils, ir, config, ir_utils, registry
from numba.core.typing.templates import (CallableTemplate, signature,
from numba.core.imputils import lower_builtin
from numba.core.extending import register_jitable
from numba.core.errors import NumbaValueError
from numba.misc.special import literal_unroll
import numba
import operator
from numba.np import numpy_support
def _stencil_wrapper(self, result, sigret, return_type, typemap, calltypes, *args):
    kernel_copy, copy_calltypes = self.copy_ir_with_calltypes(self.kernel_ir, calltypes)
    ir_utils.remove_args(kernel_copy.blocks)
    first_arg = kernel_copy.arg_names[0]
    in_cps, out_cps = ir_utils.copy_propagate(kernel_copy.blocks, typemap)
    name_var_table = ir_utils.get_name_var_table(kernel_copy.blocks)
    ir_utils.apply_copy_propagate(kernel_copy.blocks, in_cps, name_var_table, typemap, copy_calltypes)
    if 'out' in name_var_table:
        raise NumbaValueError("Cannot use the reserved word 'out' in stencil kernels.")
    sentinel_name = ir_utils.get_unused_var_name('__sentinel__', name_var_table)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('name_var_table', name_var_table, sentinel_name)
    the_array = args[0]
    if config.DEBUG_ARRAY_OPT >= 1:
        print('_stencil_wrapper', return_type, return_type.dtype, type(return_type.dtype), args)
        ir_utils.dump_blocks(kernel_copy.blocks)
    stencil_func_name = '__numba_stencil_%s_%s' % (hex(id(the_array)).replace('-', '_'), self.id)
    index_vars = []
    for i in range(the_array.ndim):
        index_var_name = ir_utils.get_unused_var_name('index' + str(i), name_var_table)
        index_vars += [index_var_name]
    out_name = ir_utils.get_unused_var_name('out', name_var_table)
    neighborhood_name = ir_utils.get_unused_var_name('neighborhood', name_var_table)
    sig_extra = ''
    if result is not None:
        sig_extra += ', {}=None'.format(out_name)
    if 'neighborhood' in dict(self.kws):
        sig_extra += ', {}=None'.format(neighborhood_name)
    standard_indexed = self.options.get('standard_indexing', [])
    if first_arg in standard_indexed:
        raise NumbaValueError('The first argument to a stencil kernel must use relative indexing, not standard indexing.')
    if len(set(standard_indexed) - set(kernel_copy.arg_names)) != 0:
        raise NumbaValueError('Standard indexing requested for an array name not present in the stencil kernel definition.')
    kernel_size, relatively_indexed = self.add_indices_to_kernel(kernel_copy, index_vars, the_array.ndim, self.neighborhood, standard_indexed, typemap, copy_calltypes)
    if self.neighborhood is None:
        self.neighborhood = kernel_size
    if config.DEBUG_ARRAY_OPT >= 1:
        print('After add_indices_to_kernel')
        ir_utils.dump_blocks(kernel_copy.blocks)
    ret_blocks = self.replace_return_with_setitem(kernel_copy.blocks, index_vars, out_name)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('After replace_return_with_setitem', ret_blocks)
        ir_utils.dump_blocks(kernel_copy.blocks)
    func_text = 'def {}({}{}):\n'.format(stencil_func_name, ','.join(kernel_copy.arg_names), sig_extra)
    ranges = []
    for i in range(the_array.ndim):
        if isinstance(kernel_size[i][0], int):
            lo = kernel_size[i][0]
            hi = kernel_size[i][1]
        else:
            lo = '{}[{}][0]'.format(neighborhood_name, i)
            hi = '{}[{}][1]'.format(neighborhood_name, i)
        ranges.append((lo, hi))
    if len(relatively_indexed) > 1:
        func_text += '    raise_if_incompatible_array_sizes(' + first_arg
        for other_array in relatively_indexed:
            if other_array != first_arg:
                func_text += ',' + other_array
        func_text += ')\n'
    shape_name = ir_utils.get_unused_var_name('full_shape', name_var_table)
    func_text += '    {} = {}.shape\n'.format(shape_name, first_arg)

    def cval_as_str(cval):
        if not np.isfinite(cval):
            if np.isnan(cval):
                return 'np.nan'
            elif np.isinf(cval):
                if cval < 0:
                    return '-np.inf'
                else:
                    return 'np.inf'
        else:
            return str(cval)
    if result is None:
        return_type_name = numpy_support.as_dtype(return_type.dtype).type.__name__
        out_init = '{} = np.empty({}, dtype=np.{})\n'.format(out_name, shape_name, return_type_name)
        if 'cval' in self.options:
            cval = self.options['cval']
            cval_ty = typing.typeof.typeof(cval)
            if not self._typingctx.can_convert(cval_ty, return_type.dtype):
                msg = 'cval type does not match stencil return type.'
                raise NumbaValueError(msg)
        else:
            cval = 0
        func_text += '    ' + out_init
        for dim in range(the_array.ndim):
            start_items = [':'] * the_array.ndim
            end_items = [':'] * the_array.ndim
            start_items[dim] = ':-{}'.format(self.neighborhood[dim][0])
            end_items[dim] = '-{}:'.format(self.neighborhood[dim][1])
            func_text += '    ' + '{}[{}] = {}\n'.format(out_name, ','.join(start_items), cval_as_str(cval))
            func_text += '    ' + '{}[{}] = {}\n'.format(out_name, ','.join(end_items), cval_as_str(cval))
    elif 'cval' in self.options:
        cval = self.options['cval']
        cval_ty = typing.typeof.typeof(cval)
        if not self._typingctx.can_convert(cval_ty, return_type.dtype):
            msg = 'cval type does not match stencil return type.'
            raise NumbaValueError(msg)
        out_init = '{}[:] = {}\n'.format(out_name, cval_as_str(cval))
        func_text += '    ' + out_init
    offset = 1
    for i in range(the_array.ndim):
        for j in range(offset):
            func_text += '    '
        func_text += 'for {} in range(-min(0,{}),{}[{}]-max(0,{})):\n'.format(index_vars[i], ranges[i][0], shape_name, i, ranges[i][1])
        offset += 1
    for j in range(offset):
        func_text += '    '
    func_text += '{} = 0\n'.format(sentinel_name)
    func_text += '    return {}\n'.format(out_name)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('new stencil func text')
        print(func_text)
    (exec(func_text) in globals(), locals())
    stencil_func = eval(stencil_func_name)
    if sigret is not None:
        pysig = utils.pysignature(stencil_func)
        sigret.pysig = pysig
    from numba.core import compiler
    stencil_ir = compiler.run_frontend(stencil_func)
    ir_utils.remove_dels(stencil_ir.blocks)
    var_table = ir_utils.get_name_var_table(stencil_ir.blocks)
    new_var_dict = {}
    reserved_names = [sentinel_name, out_name, neighborhood_name, shape_name] + kernel_copy.arg_names + index_vars
    for name, var in var_table.items():
        if not name in reserved_names:
            assert isinstance(var, ir.Var)
            new_var = var.scope.redefine(var.name, var.loc)
            new_var_dict[name] = new_var.name
    ir_utils.replace_var_names(stencil_ir.blocks, new_var_dict)
    stencil_stub_last_label = max(stencil_ir.blocks.keys()) + 1
    kernel_copy.blocks = ir_utils.add_offset_to_labels(kernel_copy.blocks, stencil_stub_last_label)
    new_label = max(kernel_copy.blocks.keys()) + 1
    ret_blocks = [x + stencil_stub_last_label for x in ret_blocks]
    if config.DEBUG_ARRAY_OPT >= 1:
        print('ret_blocks w/ offsets', ret_blocks, stencil_stub_last_label)
        print('before replace sentinel stencil_ir')
        ir_utils.dump_blocks(stencil_ir.blocks)
        print('before replace sentinel kernel_copy')
        ir_utils.dump_blocks(kernel_copy.blocks)
    for label, block in stencil_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, ir.Assign) and inst.target.name == sentinel_name:
                loc = inst.loc
                scope = block.scope
                prev_block = ir.Block(scope, loc)
                prev_block.body = block.body[:i]
                block.body = block.body[i + 1:]
                body_first_label = min(kernel_copy.blocks.keys())
                prev_block.append(ir.Jump(body_first_label, loc))
                for l, b in kernel_copy.blocks.items():
                    stencil_ir.blocks[l] = b
                stencil_ir.blocks[new_label] = block
                stencil_ir.blocks[label] = prev_block
                for ret_block in ret_blocks:
                    stencil_ir.blocks[ret_block].append(ir.Jump(new_label, loc))
                break
        else:
            continue
        break
    stencil_ir.blocks = ir_utils.rename_labels(stencil_ir.blocks)
    ir_utils.remove_dels(stencil_ir.blocks)
    assert isinstance(the_array, types.Type)
    array_types = args
    new_stencil_param_types = list(array_types)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('new_stencil_param_types', new_stencil_param_types)
        ir_utils.dump_blocks(stencil_ir.blocks)
    ir_utils.fixup_var_define_in_scope(stencil_ir.blocks)
    new_func = compiler.compile_ir(self._typingctx, self._targetctx, stencil_ir, new_stencil_param_types, None, compiler.DEFAULT_FLAGS, {})
    return new_func