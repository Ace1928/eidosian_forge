import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from numpy import (
from numpy import arange, allclose, asarray
from numpy.testing import (
def test_open_with_filename(self, tmp_path):
    tmpname = tmp_path / 'mmap'
    fp = memmap(tmpname, dtype=self.dtype, mode='w+', shape=self.shape)
    fp[:] = self.data[:]
    del fp