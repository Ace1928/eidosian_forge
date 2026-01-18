import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@lower_cast(types.UnicodeType, types.UnicodeCharSeq)
def unicode_to_unicode_charseq(context, builder, fromty, toty, val):
    uni_str = cgutils.create_struct_proxy(fromty)(context, builder, value=val)
    src1 = builder.bitcast(uni_str.data, ir.IntType(8).as_pointer())
    src2 = builder.bitcast(uni_str.data, ir.IntType(16).as_pointer())
    src4 = builder.bitcast(uni_str.data, ir.IntType(32).as_pointer())
    kind1 = builder.icmp_unsigned('==', uni_str.kind, ir.Constant(uni_str.kind.type, 1))
    kind2 = builder.icmp_unsigned('==', uni_str.kind, ir.Constant(uni_str.kind.type, 2))
    kind4 = builder.icmp_unsigned('==', uni_str.kind, ir.Constant(uni_str.kind.type, 4))
    src_length = uni_str.length
    lty = context.get_value_type(toty)
    dstint_t = ir.IntType(8 * unicode_byte_width)
    dst_ptr = cgutils.alloca_once(builder, lty)
    dst = builder.bitcast(dst_ptr, dstint_t.as_pointer())
    dst_length = ir.Constant(src_length.type, toty.count)
    is_shorter_value = builder.icmp_unsigned('<', src_length, dst_length)
    count = builder.select(is_shorter_value, src_length, dst_length)
    with builder.if_then(is_shorter_value):
        cgutils.memset(builder, dst, ir.Constant(src_length.type, toty.count * unicode_byte_width), 0)
    with builder.if_then(kind1):
        with cgutils.for_range(builder, count) as loop:
            in_ptr = builder.gep(src1, [loop.index])
            in_val = builder.zext(builder.load(in_ptr), dstint_t)
            builder.store(in_val, builder.gep(dst, [loop.index]))
    with builder.if_then(kind2):
        if unicode_byte_width >= 2:
            with cgutils.for_range(builder, count) as loop:
                in_ptr = builder.gep(src2, [loop.index])
                in_val = builder.zext(builder.load(in_ptr), dstint_t)
                builder.store(in_val, builder.gep(dst, [loop.index]))
        else:
            context.call_conv.return_user_exc(builder, ValueError, 'cannot cast 16-bit unicode_type to %s-bit %s' % (unicode_byte_width * 8, toty))
    with builder.if_then(kind4):
        if unicode_byte_width >= 4:
            with cgutils.for_range(builder, count) as loop:
                in_ptr = builder.gep(src4, [loop.index])
                in_val = builder.zext(builder.load(in_ptr), dstint_t)
                builder.store(in_val, builder.gep(dst, [loop.index]))
        else:
            context.call_conv.return_user_exc(builder, ValueError, 'cannot cast 32-bit unicode_type to %s-bit %s' % (unicode_byte_width * 8, toty))
    return builder.load(dst_ptr)