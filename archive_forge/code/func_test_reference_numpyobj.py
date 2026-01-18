import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp
from collections.abc import MutableMapping
from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
def test_reference_numpyobj(self):
    """ Object can be opened by numpy.object_ containing object ref

        Test for issue 181, issue 202.
        """
    g = self.f.create_group('test')
    dt = np.dtype([('a', 'i'), ('b', h5py.ref_dtype)])
    dset = self.f.create_dataset('test_dset', (1,), dt)
    dset[0] = (42, g.ref)
    data = dset[0]
    self.assertEqual(self.f[data[1]], g)