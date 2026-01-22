from copy import deepcopy as copy
from collections import namedtuple
import numpy as np
from .compat import filename_encode
from .datatype import Datatype
from .selections import SimpleSelection, select
from .. import h5d, h5p, h5s, h5t
class VirtualLayout:
    """Object for building a virtual dataset.

    Instantiate this class to define a virtual dataset, assign to slices of it
    (using VirtualSource objects), and then pass it to
    group.create_virtual_dataset() to add the virtual dataset to a file.

    This class does not allow access to the data; the virtual dataset must
    be created in a file before it can be used.

    shape
        A tuple giving the shape of the dataset.
    dtype
        Numpy dtype or string.
    maxshape
        The virtual dataset is resizable up to this shape. Use None for
        axes you want to be unlimited.
    filename
        The name of the destination file, if known in advance. Mappings from
        data in the same file will be stored with filename '.', allowing the
        file to be renamed later.
    """

    def __init__(self, shape, dtype, maxshape=None, filename=None):
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.dtype = dtype
        self.maxshape = (maxshape,) if isinstance(maxshape, int) else maxshape
        self._filename = filename
        self._src_filenames = set()
        self.dcpl = h5p.create(h5p.DATASET_CREATE)

    def __setitem__(self, key, source):
        sel = select(self.shape, key, dataset=None)
        _convert_space_for_key(sel.id, key)
        src_filename = self._source_file_name(source.path, self._filename)
        self.dcpl.set_virtual(sel.id, src_filename, source.name.encode('utf-8'), source.sel.id)
        if self._filename is None:
            self._src_filenames.add(src_filename)

    @staticmethod
    def _source_file_name(src_filename, dst_filename) -> bytes:
        src_filename = filename_encode(src_filename)
        if dst_filename and src_filename == filename_encode(dst_filename):
            return b'.'
        return filename_encode(src_filename)

    def _get_dcpl(self, dst_filename):
        """Get the property list containing virtual dataset mappings

        If the destination filename wasn't known when the VirtualLayout was
        created, it is handled here.
        """
        dst_filename = filename_encode(dst_filename)
        if self._filename is not None:
            if dst_filename != filename_encode(self._filename):
                raise Exception(f'{dst_filename!r} != {self._filename!r}')
            return self.dcpl
        if dst_filename in self._src_filenames:
            new_dcpl = h5p.create(h5p.DATASET_CREATE)
            for i in range(self.dcpl.get_virtual_count()):
                src_filename = self.dcpl.get_virtual_filename(i)
                new_dcpl.set_virtual(self.dcpl.get_virtual_vspace(i), self._source_file_name(src_filename, dst_filename), self.dcpl.get_virtual_dsetname(i).encode('utf-8'), self.dcpl.get_virtual_srcspace(i))
            return new_dcpl
        else:
            return self.dcpl

    def make_dataset(self, parent, name, fillvalue=None):
        """ Return a new low-level dataset identifier for a virtual dataset """
        dcpl = self._get_dcpl(parent.file.filename)
        if fillvalue is not None:
            dcpl.set_fill_value(np.array([fillvalue]))
        maxshape = self.maxshape
        if maxshape is not None:
            maxshape = tuple((m if m is not None else h5s.UNLIMITED for m in maxshape))
        virt_dspace = h5s.create_simple(self.shape, maxshape)
        if isinstance(self.dtype, Datatype):
            tid = self.dtype.id
        else:
            dtype = np.dtype(self.dtype)
            tid = h5t.py_create(dtype, logical=1)
        return h5d.create(parent.id, name=name, tid=tid, space=virt_dspace, dcpl=dcpl)