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
class TestMutableMapping(BaseGroup):
    """Tests if the registration of Group as a MutableMapping
    behaves as expected
    """

    def test_resolution(self):
        assert issubclass(Group, MutableMapping)
        grp = self.f.create_group('K')
        assert isinstance(grp, MutableMapping)

    def test_validity(self):
        """
        Test that the required functions are implemented.
        """
        Group.__getitem__
        Group.__setitem__
        Group.__delitem__
        Group.__iter__
        Group.__len__