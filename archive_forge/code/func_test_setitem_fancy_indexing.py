import pathlib
import os
import sys
import numpy as np
import platform
import pytest
import warnings
from .common import ut, TestCase
from .data_files import get_data_file_path
from h5py import File, Group, Dataset
from h5py._hl.base import is_empty_dataspace, product
from h5py import h5f, h5t
from h5py.h5py_warnings import H5pyDeprecationWarning
from h5py import version
import h5py
import h5py._hl.selections as sel
def test_setitem_fancy_indexing(writable_file):
    arr = writable_file.create_dataset('data', (5, 1000, 2), dtype=np.uint8)
    block = np.random.randint(255, size=(5, 3, 2))
    arr[:, [0, 2, 4], ...] = block