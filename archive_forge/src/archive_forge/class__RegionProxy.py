from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
class _RegionProxy:
    """
        Proxy object which handles region references.

        To create a new region reference (datasets only), use slicing syntax:

            >>> newref = obj.regionref[0:10:2]

        To determine the target dataset shape from an existing reference:

            >>> shape = obj.regionref.shape(existingref)

        where <obj> may be any object in the file. To determine the shape of
        the selection in use on the target dataset:

            >>> selection_shape = obj.regionref.selection(existingref)
    """

    def __init__(self, obj):
        self.obj = obj
        self.id = obj.id

    def __getitem__(self, args):
        if not isinstance(self.id, h5d.DatasetID):
            raise TypeError('Region references can only be made to datasets')
        from . import selections
        with phil:
            selection = selections.select(self.id.shape, args, dataset=self.obj)
            return h5r.create(self.id, b'.', h5r.DATASET_REGION, selection.id)

    def shape(self, ref):
        """ Get the shape of the target dataspace referred to by *ref*. """
        with phil:
            sid = h5r.get_region(ref, self.id)
            return sid.shape

    def selection(self, ref):
        """ Get the shape of the target dataspace selection referred to by *ref*
        """
        from . import selections
        with phil:
            sid = h5r.get_region(ref, self.id)
            return selections.guess_shape(sid)