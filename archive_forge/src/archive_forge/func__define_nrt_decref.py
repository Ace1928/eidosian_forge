from numba.core import config
from numba.core import types, cgutils
from llvmlite import ir, binding
def _define_nrt_decref(module, atomic_decr):
    """
    Implement NRT_decref in the module
    """
    fn_decref = cgutils.get_or_insert_function(module, incref_decref_ty, 'NRT_decref')
    fn_decref.attributes.add('noinline')
    calldtor = ir.Function(module, ir.FunctionType(ir.VoidType(), [_pointer_type]), name='NRT_MemInfo_call_dtor')
    builder = ir.IRBuilder(fn_decref.append_basic_block())
    [ptr] = fn_decref.args
    is_null = builder.icmp_unsigned('==', ptr, cgutils.get_null_value(ptr.type))
    with cgutils.if_unlikely(builder, is_null):
        builder.ret_void()
    builder.fence('release')
    word_ptr = builder.bitcast(ptr, atomic_decr.args[0].type)
    if config.DEBUG_NRT:
        cgutils.printf(builder, '*** NRT_Decref %zu [%p]\n', builder.load(word_ptr), ptr)
    newrefct = builder.call(atomic_decr, [word_ptr])
    refct_eq_0 = builder.icmp_unsigned('==', newrefct, ir.Constant(newrefct.type, 0))
    with cgutils.if_unlikely(builder, refct_eq_0):
        builder.fence('acquire')
        builder.call(calldtor, [ptr])
    builder.ret_void()