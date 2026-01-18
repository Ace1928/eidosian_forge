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
@ut.skipIf(NO_FS_UNICODE, 'No unicode filename support')
def test_unicode_encode(self):
    """
        Check that external links encode unicode filenames properly
        Testing issue #732
        """
    ext_filename = os.path.join(mkdtemp(), u'α.hdf5')
    with File(ext_filename, 'w') as ext_file:
        ext_file.create_group('external')
    self.f['ext'] = ExternalLink(ext_filename, '/external')