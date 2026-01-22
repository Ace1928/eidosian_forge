from numba.core import config
from numba.core import types, cgutils
from llvmlite import ir, binding
def _define_nrt_meminfo_data(module):
    """
    Implement NRT_MemInfo_data_fast in the module.  This allows LLVM
    to inline lookup of the data pointer.
    """
    fn = cgutils.get_or_insert_function(module, meminfo_data_ty, 'NRT_MemInfo_data_fast')
    builder = ir.IRBuilder(fn.append_basic_block())
    [ptr] = fn.args
    struct_ptr = builder.bitcast(ptr, _meminfo_struct_type.as_pointer())
    data_ptr = builder.load(cgutils.gep(builder, struct_ptr, 0, 3))
    builder.ret(data_ptr)