from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
class HLObject(CommonStateObject):
    """
        Base class for high-level interface objects.
    """

    @property
    def file(self):
        """ Return a File instance associated with this object """
        from . import files
        with phil:
            return files.File(self.id)

    @property
    @with_phil
    def name(self):
        """ Return the full name of this object.  None if anonymous. """
        return self._d(h5i.get_name(self.id))

    @property
    @with_phil
    def parent(self):
        """Return the parent group of this object.

        This is always equivalent to obj.file[posixpath.dirname(obj.name)].
        ValueError if this object is anonymous.
        """
        if self.name is None:
            raise ValueError('Parent of an anonymous object is undefined')
        return self.file[posixpath.dirname(self.name)]

    @property
    @with_phil
    def id(self):
        """ Low-level identifier appropriate for this object """
        return self._id

    @property
    @with_phil
    def ref(self):
        """ An (opaque) HDF5 reference to this object """
        return h5r.create(self.id, b'.', h5r.OBJECT)

    @property
    @with_phil
    def regionref(self):
        """Create a region reference (Datasets only).

        The syntax is regionref[<slices>]. For example, dset.regionref[...]
        creates a region reference in which the whole dataset is selected.

        Can also be used to determine the shape of the referenced dataset
        (via .shape property), or the shape of the selection (via the
        .selection property).
        """
        return _RegionProxy(self)

    @property
    def attrs(self):
        """ Attributes attached to this object """
        from . import attrs
        with phil:
            return attrs.AttributeManager(self)

    @with_phil
    def __init__(self, oid):
        """ Setup this object, given its low-level identifier """
        self._id = oid

    @with_phil
    def __hash__(self):
        return hash(self.id)

    @with_phil
    def __eq__(self, other):
        if hasattr(other, 'id'):
            return self.id == other.id
        return NotImplemented

    def __bool__(self):
        with phil:
            return bool(self.id)
    __nonzero__ = __bool__

    def __getnewargs__(self):
        """Disable pickle.

        Handles for HDF5 objects can't be reliably deserialised, because the
        recipient may not have access to the same files. So we do this to
        fail early.

        If you really want to pickle h5py objects and can live with some
        limitations, look at the h5pickle project on PyPI.
        """
        raise TypeError('h5py objects cannot be pickled')

    def __getstate__(self):
        raise TypeError('h5py objects cannot be pickled')