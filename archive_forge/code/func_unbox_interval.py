import unittest
@unbox(IntervalType)
def unbox_interval(typ, obj, c):
    """
            Convert a Interval object to a native interval structure.
            """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    with ExitStack() as stack:
        lo_obj = c.pyapi.object_getattr_string(obj, 'lo')
        with cgutils.early_exit_if_null(c.builder, stack, lo_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        lo_native = c.unbox(types.float64, lo_obj)
        c.pyapi.decref(lo_obj)
        with cgutils.early_exit_if(c.builder, stack, lo_native.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        hi_obj = c.pyapi.object_getattr_string(obj, 'hi')
        with cgutils.early_exit_if_null(c.builder, stack, hi_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        hi_native = c.unbox(types.float64, hi_obj)
        c.pyapi.decref(hi_obj)
        with cgutils.early_exit_if(c.builder, stack, hi_native.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        interval.lo = lo_native.value
        interval.hi = hi_native.value
    return NativeValue(interval._getvalue(), is_error=c.builder.load(is_error_ptr))