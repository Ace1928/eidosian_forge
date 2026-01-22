import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def _lower_parfor_parallel_std(lowerer, parfor):
    """Lowerer that handles LLVM code generation for parfor.
    This function lowers a parfor IR node to LLVM.
    The general approach is as follows:
    1) The code from the parfor's init block is lowered normally
       in the context of the current function.
    2) The body of the parfor is transformed into a gufunc function.
    3) Code is inserted into the main function that calls do_scheduling
       to divide the iteration space for each thread, allocates
       reduction arrays, calls the gufunc function, and then invokes
       the reduction function across the reduction arrays to produce
       the final reduction values.
    """
    from numba.np.ufunc.parallel import get_thread_count
    ensure_parallel_support()
    typingctx = lowerer.context.typing_context
    targetctx = lowerer.context
    builder = lowerer.builder
    orig_typemap = lowerer.fndesc.typemap
    lowerer.fndesc.typemap = copy.copy(orig_typemap)
    if config.DEBUG_ARRAY_OPT:
        print('lowerer.fndesc', lowerer.fndesc, type(lowerer.fndesc))
    typemap = lowerer.fndesc.typemap
    varmap = lowerer.varmap
    if config.DEBUG_ARRAY_OPT:
        print('_lower_parfor_parallel')
        parfor.dump()
    loc = parfor.init_block.loc
    scope = parfor.init_block.scope
    if config.DEBUG_ARRAY_OPT:
        print('init_block = ', parfor.init_block, ' ', type(parfor.init_block))
    for instr in parfor.init_block.body:
        if config.DEBUG_ARRAY_OPT:
            print('lower init_block instr = ', instr)
        lowerer.lower_inst(instr)
    for racevar in parfor.races:
        if racevar not in varmap:
            rvtyp = typemap[racevar]
            rv = ir.Var(scope, racevar, loc)
            lowerer._alloca_var(rv.name, rvtyp)
    alias_map = {}
    arg_aliases = {}
    numba.parfors.parfor.find_potential_aliases_parfor(parfor, parfor.params, typemap, lowerer.func_ir, alias_map, arg_aliases)
    if config.DEBUG_ARRAY_OPT:
        print('alias_map', alias_map)
        print('arg_aliases', arg_aliases)
    assert parfor.params is not None
    parfor_output_arrays = numba.parfors.parfor.get_parfor_outputs(parfor, parfor.params)
    parfor_redvars, parfor_reddict = (parfor.redvars, parfor.reddict)
    if config.DEBUG_ARRAY_OPT:
        print('parfor_redvars:', parfor_redvars)
        print('parfor_reddict:', parfor_reddict)
    nredvars = len(parfor_redvars)
    redarrs = {}
    to_cleanup = []
    if nredvars > 0:
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        pfbdr = ParforLoweringBuilder(lowerer=lowerer, scope=scope, loc=loc)
        get_num_threads = pfbdr.bind_global_function(fobj=numba.np.ufunc.parallel._iget_num_threads, ftype=get_global_func_typ(numba.np.ufunc.parallel._iget_num_threads), args=())
        num_threads_var = pfbdr.assign(rhs=pfbdr.call(get_num_threads, args=[]), typ=types.intp, name='num_threads_var')
        for i in range(nredvars):
            red_name = parfor_redvars[i]
            redvar_typ = lowerer.fndesc.typemap[red_name]
            redvar = ir.Var(scope, red_name, loc)
            redarrvar_typ = redtyp_to_redarraytype(redvar_typ)
            reddtype = redarrvar_typ.dtype
            if config.DEBUG_ARRAY_OPT:
                print('reduction_info', red_name, redvar_typ, redarrvar_typ, reddtype, types.DType(reddtype), num_threads_var, type(num_threads_var))
            if isinstance(redvar_typ, types.npytypes.Array):
                redarrdim = redvar_typ.ndim + 1
            else:
                redarrdim = 1
            glbl_np_empty = pfbdr.bind_global_function(fobj=np.empty, ftype=get_np_ufunc_typ(np.empty), args=(types.UniTuple(types.intp, redarrdim),), kws={'dtype': types.DType(reddtype)})
            size_var_list = [num_threads_var]
            if isinstance(redvar_typ, types.npytypes.Array):
                redshape_var = pfbdr.assign(rhs=ir.Expr.getattr(redvar, 'shape', loc), typ=types.UniTuple(types.intp, redvar_typ.ndim), name='redarr_shape')
                for j in range(redvar_typ.ndim):
                    onedimvar = pfbdr.assign(rhs=ir.Expr.static_getitem(redshape_var, j, None, loc), typ=types.intp, name='redshapeonedim')
                    size_var_list.append(onedimvar)
            size_var = pfbdr.make_tuple_variable(size_var_list, name='tuple_size_var')
            cval = pfbdr._typingctx.resolve_value_type(reddtype)
            dt = pfbdr.make_const_variable(cval=cval, typ=types.DType(reddtype))
            empty_call = pfbdr.call(glbl_np_empty, args=[size_var, dt])
            redarr_var = pfbdr.assign(rhs=empty_call, typ=redarrvar_typ, name='redarr')
            redarrs[redvar.name] = redarr_var
            to_cleanup.append(redarr_var)
            init_val = parfor_reddict[red_name].init_val
            if init_val is not None:
                if isinstance(redvar_typ, types.npytypes.Array):
                    full_func_node = pfbdr.bind_global_function(fobj=np.full, ftype=get_np_ufunc_typ(np.full), args=(types.UniTuple(types.intp, redvar_typ.ndim), reddtype), kws={'dtype': types.DType(reddtype)})
                    init_val_var = pfbdr.make_const_variable(cval=init_val, typ=reddtype, name='init_val')
                    full_call = pfbdr.call(full_func_node, args=[redshape_var, init_val_var, dt])
                    redtoset = pfbdr.assign(rhs=full_call, typ=redvar_typ, name='redtoset')
                    to_cleanup.append(redtoset)
                else:
                    redtoset = pfbdr.make_const_variable(cval=init_val, typ=reddtype, name='redtoset')
            else:
                redtoset = redvar
                if config.DEBUG_ARRAY_OPT_RUNTIME:
                    res_print_str = 'res_print1 for redvar ' + str(redvar) + ':'
                    strconsttyp = types.StringLiteral(res_print_str)
                    lhs = pfbdr.make_const_variable(cval=res_print_str, typ=strconsttyp, name='str_const')
                    res_print = ir.Print(args=[lhs, redvar], vararg=None, loc=loc)
                    lowerer.fndesc.calltypes[res_print] = signature(types.none, typemap[lhs.name], typemap[redvar.name])
                    print('res_print_redvar', res_print)
                    lowerer.lower_inst(res_print)
            num_thread_type = typemap[num_threads_var.name]
            ntllvm_type = targetctx.get_value_type(num_thread_type)
            alloc_loop_var = cgutils.alloca_once(builder, ntllvm_type)
            numba_ir_loop_index_var = scope.redefine('$loop_index', loc)
            typemap[numba_ir_loop_index_var.name] = num_thread_type
            lowerer.varmap[numba_ir_loop_index_var.name] = alloc_loop_var
            with cgutils.for_range(builder, lowerer.loadvar(num_threads_var.name), intp=ntllvm_type) as loop:
                builder.store(loop.index, alloc_loop_var)
                pfbdr.setitem(obj=redarr_var, index=numba_ir_loop_index_var, val=redtoset)
    flags = parfor.flags.copy()
    flags.error_model = 'numpy'
    index_var_typ = typemap[parfor.loop_nests[0].index_variable.name]
    for l in parfor.loop_nests[1:]:
        assert typemap[l.index_variable.name] == index_var_typ
    numba.parfors.parfor.sequential_parfor_lowering = True
    try:
        func, func_args, func_sig, func_arg_types, exp_name_to_tuple_var = _create_gufunc_for_parfor_body(lowerer, parfor, typemap, typingctx, targetctx, flags, {}, bool(alias_map), index_var_typ, parfor.races)
    finally:
        numba.parfors.parfor.sequential_parfor_lowering = False
    func_args = ['sched'] + func_args
    num_reductions = len(parfor_redvars)
    num_inputs = len(func_args) - len(parfor_output_arrays) - num_reductions
    if config.DEBUG_ARRAY_OPT:
        print('func_args = ', func_args)
        print('num_inputs = ', num_inputs)
        print('parfor_outputs = ', parfor_output_arrays)
        print('parfor_redvars = ', parfor_redvars)
        print('num_reductions = ', num_reductions)
    gu_signature = _create_shape_signature(parfor.get_shape_classes, num_inputs, num_reductions, func_args, func_sig, parfor.races, typemap)
    if config.DEBUG_ARRAY_OPT:
        print('gu_signature = ', gu_signature)
    loop_ranges = [(l.start, l.stop, l.step) for l in parfor.loop_nests]
    if config.DEBUG_ARRAY_OPT:
        print('loop_nests = ', parfor.loop_nests)
        print('loop_ranges = ', loop_ranges)
    call_parallel_gufunc(lowerer, func, gu_signature, func_sig, func_args, func_arg_types, loop_ranges, parfor_redvars, parfor_reddict, redarrs, parfor.init_block, index_var_typ, parfor.races, exp_name_to_tuple_var)
    if nredvars > 0:
        _parfor_lowering_finalize_reduction(parfor, redarrs, lowerer, parfor_reddict, num_threads_var)
    for v in to_cleanup:
        lowerer.lower_inst(ir.Del(v.name, loc=loc))
    lowerer.fndesc.typemap = orig_typemap
    if config.DEBUG_ARRAY_OPT:
        print('_lower_parfor_parallel done')