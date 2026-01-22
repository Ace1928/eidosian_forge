from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
class StructModel(CompositeModel):
    _value_type = None
    _data_type = None

    def __init__(self, dmm, fe_type, members):
        super(StructModel, self).__init__(dmm, fe_type)
        if members:
            self._fields, self._members = zip(*members)
        else:
            self._fields = self._members = ()
        self._models = tuple([self._dmm.lookup(t) for t in self._members])

    def get_member_fe_type(self, name):
        """
        StructModel-specific: get the Numba type of the field named *name*.
        """
        pos = self.get_field_position(name)
        return self._members[pos]

    def get_value_type(self):
        if self._value_type is None:
            self._value_type = ir.LiteralStructType([t.get_value_type() for t in self._models])
        return self._value_type

    def get_data_type(self):
        if self._data_type is None:
            self._data_type = ir.LiteralStructType([t.get_data_type() for t in self._models])
        return self._data_type

    def get_argument_type(self):
        return tuple([t.get_argument_type() for t in self._models])

    def get_return_type(self):
        return self.get_data_type()

    def _as(self, methname, builder, value):
        extracted = []
        for i, dm in enumerate(self._models):
            extracted.append(getattr(dm, methname)(builder, self.get(builder, value, i)))
        return tuple(extracted)

    def _from(self, methname, builder, value):
        struct = ir.Constant(self.get_value_type(), ir.Undefined)
        for i, (dm, val) in enumerate(zip(self._models, value)):
            v = getattr(dm, methname)(builder, val)
            struct = self.set(builder, struct, v, i)
        return struct

    def as_data(self, builder, value):
        """
        Converts the LLVM struct in `value` into a representation suited for
        storing into arrays.

        Note
        ----
        Current implementation rarely changes how types are represented for
        "value" and "data".  This is usually a pointless rebuild of the
        immutable LLVM struct value.  Luckily, LLVM optimization removes all
        redundancy.

        Sample usecase: Structures nested with pointers to other structures
        that can be serialized into  a flat representation when storing into
        array.
        """
        elems = self._as('as_data', builder, value)
        struct = ir.Constant(self.get_data_type(), ir.Undefined)
        for i, el in enumerate(elems):
            struct = builder.insert_value(struct, el, [i])
        return struct

    def from_data(self, builder, value):
        """
        Convert from "data" representation back into "value" representation.
        Usually invoked when loading from array.

        See notes in `as_data()`
        """
        vals = [builder.extract_value(value, [i]) for i in range(len(self._members))]
        return self._from('from_data', builder, vals)

    def load_from_data_pointer(self, builder, ptr, align=None):
        values = []
        for i, model in enumerate(self._models):
            elem_ptr = cgutils.gep_inbounds(builder, ptr, 0, i)
            val = model.load_from_data_pointer(builder, elem_ptr, align)
            values.append(val)
        struct = ir.Constant(self.get_value_type(), ir.Undefined)
        for i, val in enumerate(values):
            struct = self.set(builder, struct, val, i)
        return struct

    def as_argument(self, builder, value):
        return self._as('as_argument', builder, value)

    def from_argument(self, builder, value):
        return self._from('from_argument', builder, value)

    def as_return(self, builder, value):
        elems = self._as('as_data', builder, value)
        struct = ir.Constant(self.get_data_type(), ir.Undefined)
        for i, el in enumerate(elems):
            struct = builder.insert_value(struct, el, [i])
        return struct

    def from_return(self, builder, value):
        vals = [builder.extract_value(value, [i]) for i in range(len(self._members))]
        return self._from('from_data', builder, vals)

    def get(self, builder, val, pos):
        """Get a field at the given position or the fieldname

        Args
        ----
        builder:
            LLVM IRBuilder
        val:
            value to be inserted
        pos: int or str
            field index or field name

        Returns
        -------
        Extracted value
        """
        if isinstance(pos, str):
            pos = self.get_field_position(pos)
        return builder.extract_value(val, [pos], name='extracted.' + self._fields[pos])

    def set(self, builder, stval, val, pos):
        """Set a field at the given position or the fieldname

        Args
        ----
        builder:
            LLVM IRBuilder
        stval:
            LLVM struct value
        val:
            value to be inserted
        pos: int or str
            field index or field name

        Returns
        -------
        A new LLVM struct with the value inserted
        """
        if isinstance(pos, str):
            pos = self.get_field_position(pos)
        return builder.insert_value(stval, val, [pos], name='inserted.' + self._fields[pos])

    def get_field_position(self, field):
        try:
            return self._fields.index(field)
        except ValueError:
            raise KeyError('%s does not have a field named %r' % (self.__class__.__name__, field))

    @property
    def field_count(self):
        return len(self._fields)

    def get_type(self, pos):
        """Get the frontend type (numba type) of a field given the position
         or the fieldname

        Args
        ----
        pos: int or str
            field index or field name
        """
        if isinstance(pos, str):
            pos = self.get_field_position(pos)
        return self._members[pos]

    def get_model(self, pos):
        """
        Get the datamodel of a field given the position or the fieldname.

        Args
        ----
        pos: int or str
            field index or field name
        """
        return self._models[pos]

    def traverse(self, builder):

        def getter(k, value):
            if value.type != self.get_value_type():
                args = (self.get_value_type(), value.type)
                raise TypeError('expecting {0} but got {1}'.format(*args))
            return self.get(builder, value, k)
        return [(self.get_type(k), partial(getter, k)) for k in self._fields]

    def inner_models(self):
        return self._models