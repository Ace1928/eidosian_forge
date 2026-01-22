import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
class Structure(object):
    """
    A high-level object wrapping a alloca'ed LLVM structure, including
    named fields and attribute access.
    """

    def __init__(self, context, builder, value=None, ref=None, cast_ref=False):
        self._type = context.get_struct_type(self)
        self._context = context
        self._builder = builder
        if ref is None:
            self._value = alloca_once(builder, self._type, zfill=True)
            if value is not None:
                assert not is_pointer(value.type)
                assert value.type == self._type, (value.type, self._type)
                builder.store(value, self._value)
        else:
            assert value is None
            assert is_pointer(ref.type)
            if self._type != ref.type.pointee:
                if cast_ref:
                    ref = builder.bitcast(ref, self._type.as_pointer())
                else:
                    raise TypeError('mismatching pointer type: got %s, expected %s' % (ref.type.pointee, self._type))
            self._value = ref
        self._namemap = {}
        self._fdmap = []
        self._typemap = []
        base = int32_t(0)
        for i, (k, tp) in enumerate(self._fields):
            self._namemap[k] = i
            self._fdmap.append((base, int32_t(i)))
            self._typemap.append(tp)

    def _get_ptr_by_index(self, index):
        ptr = self._builder.gep(self._value, self._fdmap[index], inbounds=True)
        return ptr

    def _get_ptr_by_name(self, attrname):
        return self._get_ptr_by_index(self._namemap[attrname])

    def __getattr__(self, field):
        """
        Load the LLVM value of the named *field*.
        """
        if not field.startswith('_'):
            return self[self._namemap[field]]
        else:
            raise AttributeError(field)

    def __setattr__(self, field, value):
        """
        Store the LLVM *value* into the named *field*.
        """
        if field.startswith('_'):
            return super(Structure, self).__setattr__(field, value)
        self[self._namemap[field]] = value

    def __getitem__(self, index):
        """
        Load the LLVM value of the field at *index*.
        """
        return self._builder.load(self._get_ptr_by_index(index))

    def __setitem__(self, index, value):
        """
        Store the LLVM *value* into the field at *index*.
        """
        ptr = self._get_ptr_by_index(index)
        if ptr.type.pointee != value.type:
            fmt = 'Type mismatch: __setitem__(%d, ...) expected %r but got %r'
            raise AssertionError(fmt % (index, str(ptr.type.pointee), str(value.type)))
        self._builder.store(value, ptr)

    def __len__(self):
        """
        Return the number of fields.
        """
        return len(self._namemap)

    def _getpointer(self):
        """
        Return the LLVM pointer to the underlying structure.
        """
        return self._value

    def _getvalue(self):
        """
        Load and return the value of the underlying LLVM structure.
        """
        return self._builder.load(self._value)

    def _setvalue(self, value):
        """Store the value in this structure"""
        assert not is_pointer(value.type)
        assert value.type == self._type, (value.type, self._type)
        self._builder.store(value, self._value)