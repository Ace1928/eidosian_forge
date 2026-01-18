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
def test_setdefault_no_default(self):
    """
        .setdefault gets None if group doesn't exist, but as None isn't defined
        as data for a dataset, this should raise a TypeError.
        """
    with self.assertRaises(TypeError):
        self.group.setdefault('e')