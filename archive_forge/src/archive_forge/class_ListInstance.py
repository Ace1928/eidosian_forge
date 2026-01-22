import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
class ListInstance(_ListPayloadMixin):

    def __init__(self, context, builder, list_type, list_val):
        self._context = context
        self._builder = builder
        self._ty = list_type
        self._list = context.make_helper(builder, list_type, list_val)
        self._itemsize = get_itemsize(context, list_type)
        self._datamodel = context.data_model_manager[list_type.dtype]

    @property
    def dtype(self):
        return self._ty.dtype

    @property
    def _payload(self):
        return get_list_payload(self._context, self._builder, self._ty, self._list)

    @property
    def parent(self):
        return self._list.parent

    @parent.setter
    def parent(self, value):
        self._list.parent = value

    @property
    def value(self):
        return self._list._getvalue()

    @property
    def meminfo(self):
        return self._list.meminfo

    def set_dirty(self, val):
        if self._ty.reflected:
            self._payload.dirty = cgutils.true_bit if val else cgutils.false_bit

    def clear_value(self, idx):
        """Remove the value at the location
        """
        self.decref_value(self.getitem(idx))
        self.zfill(idx, self._builder.add(idx, idx.type(1)))

    def setitem(self, idx, val, incref, decref_old_value=True):
        if decref_old_value:
            self.decref_value(self.getitem(idx))
        ptr = self._gep(idx)
        data_item = self._datamodel.as_data(self._builder, val)
        self._builder.store(data_item, ptr)
        self.set_dirty(True)
        if incref:
            self.incref_value(val)

    def inititem(self, idx, val, incref=True):
        ptr = self._gep(idx)
        data_item = self._datamodel.as_data(self._builder, val)
        self._builder.store(data_item, ptr)
        if incref:
            self.incref_value(val)

    def zfill(self, start, stop):
        """Zero-fill the memory at index *start* to *stop*

        *stop* MUST not be smaller than *start*.
        """
        builder = self._builder
        base = self._gep(start)
        end = self._gep(stop)
        intaddr_t = self._context.get_value_type(types.intp)
        size = builder.sub(builder.ptrtoint(end, intaddr_t), builder.ptrtoint(base, intaddr_t))
        cgutils.memset(builder, base, size, ir.IntType(8)(0))

    @classmethod
    def allocate_ex(cls, context, builder, list_type, nitems):
        """
        Allocate a ListInstance with its storage.
        Return a (ok, instance) tuple where *ok* is a LLVM boolean and
        *instance* is a ListInstance object (the object's contents are
        only valid when *ok* is true).
        """
        intp_t = context.get_value_type(types.intp)
        if isinstance(nitems, int):
            nitems = ir.Constant(intp_t, nitems)
        payload_type = context.get_data_type(types.ListPayload(list_type))
        payload_size = context.get_abi_sizeof(payload_type)
        itemsize = get_itemsize(context, list_type)
        payload_size -= itemsize
        ok = cgutils.alloca_once_value(builder, cgutils.true_bit)
        self = cls(context, builder, list_type, None)
        allocsize, ovf = cgutils.muladd_with_overflow(builder, nitems, ir.Constant(intp_t, itemsize), ir.Constant(intp_t, payload_size))
        with builder.if_then(ovf, likely=False):
            builder.store(cgutils.false_bit, ok)
        with builder.if_then(builder.load(ok), likely=True):
            meminfo = context.nrt.meminfo_new_varsize_dtor_unchecked(builder, size=allocsize, dtor=self.get_dtor())
            with builder.if_else(cgutils.is_null(builder, meminfo), likely=False) as (if_error, if_ok):
                with if_error:
                    builder.store(cgutils.false_bit, ok)
                with if_ok:
                    self._list.meminfo = meminfo
                    self._list.parent = context.get_constant_null(types.pyobject)
                    self._payload.allocated = nitems
                    self._payload.size = ir.Constant(intp_t, 0)
                    self._payload.dirty = cgutils.false_bit
                    self.zfill(self.size.type(0), nitems)
        return (builder.load(ok), self)

    def define_dtor(self):
        """Define the destructor if not already defined"""
        context = self._context
        builder = self._builder
        mod = builder.module
        fnty = ir.FunctionType(ir.VoidType(), [cgutils.voidptr_t])
        fn = cgutils.get_or_insert_function(mod, fnty, '.dtor.list.{}'.format(self.dtype))
        if not fn.is_declaration:
            return fn
        fn.linkage = 'linkonce_odr'
        builder = ir.IRBuilder(fn.append_basic_block())
        base_ptr = fn.args[0]
        payload = ListPayloadAccessor(context, builder, self._ty, base_ptr)
        intp = payload.size.type
        with cgutils.for_range_slice(builder, start=intp(0), stop=payload.size, step=intp(1), intp=intp) as (idx, _):
            val = payload.getitem(idx)
            context.nrt.decref(builder, self.dtype, val)
        builder.ret_void()
        return fn

    def get_dtor(self):
        """"Get the element dtor function pointer as void pointer.

        It's safe to be called multiple times.
        """
        dtor = self.define_dtor()
        dtor_fnptr = self._builder.bitcast(dtor, cgutils.voidptr_t)
        return dtor_fnptr

    @classmethod
    def allocate(cls, context, builder, list_type, nitems):
        """
        Allocate a ListInstance with its storage.  Same as allocate_ex(),
        but return an initialized *instance*.  If allocation failed,
        control is transferred to the caller using the target's current
        call convention.
        """
        ok, self = cls.allocate_ex(context, builder, list_type, nitems)
        with builder.if_then(builder.not_(ok), likely=False):
            context.call_conv.return_user_exc(builder, MemoryError, ('cannot allocate list',))
        return self

    @classmethod
    def from_meminfo(cls, context, builder, list_type, meminfo):
        """
        Allocate a new list instance pointing to an existing payload
        (a meminfo pointer).
        Note the parent field has to be filled by the caller.
        """
        self = cls(context, builder, list_type, None)
        self._list.meminfo = meminfo
        self._list.parent = context.get_constant_null(types.pyobject)
        context.nrt.incref(builder, list_type, self.value)
        return self

    def resize(self, new_size):
        """
        Ensure the list is properly sized for the new size.
        """

        def _payload_realloc(new_allocated):
            payload_type = context.get_data_type(types.ListPayload(self._ty))
            payload_size = context.get_abi_sizeof(payload_type)
            payload_size -= itemsize
            allocsize, ovf = cgutils.muladd_with_overflow(builder, new_allocated, ir.Constant(intp_t, itemsize), ir.Constant(intp_t, payload_size))
            with builder.if_then(ovf, likely=False):
                context.call_conv.return_user_exc(builder, MemoryError, ('cannot resize list',))
            ptr = context.nrt.meminfo_varsize_realloc_unchecked(builder, self._list.meminfo, size=allocsize)
            cgutils.guard_memory_error(context, builder, ptr, 'cannot resize list')
            self._payload.allocated = new_allocated
        context = self._context
        builder = self._builder
        intp_t = new_size.type
        itemsize = get_itemsize(context, self._ty)
        allocated = self._payload.allocated
        two = ir.Constant(intp_t, 2)
        eight = ir.Constant(intp_t, 8)
        is_too_small = builder.icmp_signed('<', allocated, new_size)
        is_too_large = builder.icmp_signed('>', builder.ashr(allocated, two), new_size)
        with builder.if_then(is_too_large, likely=False):
            _payload_realloc(new_size)
        with builder.if_then(is_too_small, likely=False):
            new_allocated = builder.add(eight, builder.add(new_size, builder.ashr(new_size, two)))
            _payload_realloc(new_allocated)
            self.zfill(self.size, new_allocated)
        self._payload.size = new_size
        self.set_dirty(True)

    def move(self, dest_idx, src_idx, count):
        """
        Move `count` elements from `src_idx` to `dest_idx`.
        """
        dest_ptr = self._gep(dest_idx)
        src_ptr = self._gep(src_idx)
        cgutils.raw_memmove(self._builder, dest_ptr, src_ptr, count, itemsize=self._itemsize)
        self.set_dirty(True)