import numpy
import uuid
from .. import h5, h5s, h5t, h5a, h5p
from . import base
from .base import phil, with_phil, Empty, is_empty_dataspace, product
from .datatype import Datatype
class AttributeManager(base.MutableMappingHDF5, base.CommonStateObject):
    """
        Allows dictionary-style access to an HDF5 object's attributes.

        These are created exclusively by the library and are available as
        a Python attribute at <object>.attrs

        Like Group objects, attributes provide a minimal dictionary-
        style interface.  Anything which can be reasonably converted to a
        Numpy array or Numpy scalar can be stored.

        Attributes are automatically created on assignment with the
        syntax <obj>.attrs[name] = value, with the HDF5 type automatically
        deduced from the value.  Existing attributes are overwritten.

        To modify an existing attribute while preserving its type, use the
        method modify().  To specify an attribute of a particular type and
        shape, use create().
    """

    def __init__(self, parent):
        """ Private constructor.
        """
        self._id = parent.id

    @with_phil
    def __getitem__(self, name):
        """ Read the value of an attribute.
        """
        attr = h5a.open(self._id, self._e(name))
        shape = attr.shape
        if shape is None:
            return Empty(attr.dtype)
        dtype = attr.dtype
        htype = h5t.py_create(dtype)
        if dtype.subdtype is not None:
            subdtype, subshape = dtype.subdtype
            shape = attr.shape + subshape
            dtype = subdtype
        arr = numpy.zeros(shape, dtype=dtype, order='C')
        attr.read(arr, mtype=htype)
        string_info = h5t.check_string_dtype(dtype)
        if string_info and string_info.length is None:
            arr = numpy.array([b.decode('utf-8', 'surrogateescape') for b in arr.flat], dtype=dtype).reshape(arr.shape)
        if arr.ndim == 0:
            return arr[()]
        return arr

    def get_id(self, name):
        """Get a low-level AttrID object for the named attribute.
        """
        return h5a.open(self._id, self._e(name))

    @with_phil
    def __setitem__(self, name, value):
        """ Set a new attribute, overwriting any existing attribute.

        The type and shape of the attribute are determined from the data.  To
        use a specific type or shape, or to preserve the type of an attribute,
        use the methods create() and modify().
        """
        self.create(name, data=value)

    @with_phil
    def __delitem__(self, name):
        """ Delete an attribute (which must already exist). """
        h5a.delete(self._id, self._e(name))

    def create(self, name, data, shape=None, dtype=None):
        """ Create a new attribute, overwriting any existing attribute.

        name
            Name of the new attribute (required)
        data
            An array to initialize the attribute (required)
        shape
            Shape of the attribute.  Overrides data.shape if both are
            given, in which case the total number of points must be unchanged.
        dtype
            Data type of the attribute.  Overrides data.dtype if both
            are given.
        """
        name = self._e(name)
        with phil:
            if not isinstance(data, Empty):
                data = base.array_for_new_object(data, specified_dtype=dtype)
            if shape is None:
                shape = data.shape
            elif isinstance(shape, int):
                shape = (shape,)
            use_htype = None
            if isinstance(dtype, Datatype):
                use_htype = dtype.id
                dtype = dtype.dtype
            elif dtype is None:
                dtype = data.dtype
            else:
                dtype = numpy.dtype(dtype)
            original_dtype = dtype
            if dtype.subdtype is not None:
                subdtype, subshape = dtype.subdtype
                if shape[-len(subshape):] != subshape:
                    raise ValueError('Array dtype shape %s is incompatible with data shape %s' % (subshape, shape))
                shape = shape[0:len(shape) - len(subshape)]
                dtype = subdtype
            else:
                if shape is not None and product(shape) != product(data.shape):
                    raise ValueError('Shape of new attribute conflicts with shape of data')
                if shape != data.shape:
                    data = data.reshape(shape)
            if not isinstance(data, Empty):
                data = numpy.asarray(data, dtype=dtype)
            if use_htype is None:
                htype = h5t.py_create(original_dtype, logical=True)
                htype2 = h5t.py_create(original_dtype)
            else:
                htype = use_htype
                htype2 = None
            if isinstance(data, Empty):
                space = h5s.create(h5s.NULL)
            else:
                space = h5s.create_simple(shape)
            if h5a.exists(self._id, name):
                h5a.delete(self._id, name)
            attr = h5a.create(self._id, name, htype, space)
            try:
                if not isinstance(data, Empty):
                    attr.write(data, mtype=htype2)
            except:
                attr.close()
                h5a.delete(self._id, name)
                raise
            attr.close()

    def modify(self, name, value):
        """ Change the value of an attribute while preserving its type.

        Differs from __setitem__ in that if the attribute already exists, its
        type is preserved.  This can be very useful for interacting with
        externally generated files.

        If the attribute doesn't exist, it will be automatically created.
        """
        with phil:
            if not name in self:
                self[name] = value
            else:
                attr = h5a.open(self._id, self._e(name))
                if is_empty_dataspace(attr):
                    raise OSError("Empty attributes can't be modified")
                dt = None if isinstance(value, numpy.ndarray) else attr.dtype
                value = numpy.asarray(value, order='C', dtype=dt)
                if value.shape != attr.shape and (not (value.size == 1 and product(attr.shape) == 1)):
                    raise TypeError('Shape of data is incompatible with existing attribute')
                attr.write(value)

    @with_phil
    def __len__(self):
        """ Number of attributes attached to the object. """
        return h5a.get_num_attrs(self._id)

    def __iter__(self):
        """ Iterate over the names of attributes. """
        with phil:
            attrlist = []

            def iter_cb(name, *args):
                """ Callback to gather attribute names """
                attrlist.append(self._d(name))
            cpl = self._id.get_create_plist()
            crt_order = cpl.get_attr_creation_order()
            cpl.close()
            if crt_order & h5p.CRT_ORDER_TRACKED:
                idx_type = h5.INDEX_CRT_ORDER
            else:
                idx_type = h5.INDEX_NAME
            h5a.iterate(self._id, iter_cb, index_type=idx_type)
        for name in attrlist:
            yield name

    @with_phil
    def __contains__(self, name):
        """ Determine if an attribute exists, by name. """
        return h5a.exists(self._id, self._e(name))

    @with_phil
    def __repr__(self):
        if not self._id:
            return '<Attributes of closed HDF5 object>'
        return '<Attributes of HDF5 object at %s>' % id(self._id)