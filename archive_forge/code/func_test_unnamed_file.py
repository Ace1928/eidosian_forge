import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from numpy import (
from numpy import arange, allclose, asarray
from numpy.testing import (
def test_unnamed_file(self):
    with TemporaryFile() as f:
        fp = memmap(f, dtype=self.dtype, shape=self.shape)
        del fp