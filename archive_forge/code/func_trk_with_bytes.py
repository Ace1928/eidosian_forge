import copy
import os
import sys
import unittest
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arr_dict_equal, clear_and_catch_warnings, data_path, error_warnings
from .. import trk as trk_module
from ..header import Field
from ..tractogram import Tractogram
from ..tractogram_file import HeaderError, HeaderWarning
from ..trk import (
from .test_tractogram import assert_tractogram_equal
def trk_with_bytes(self, trk_key='simple_trk_fname', endian='<'):
    """Return example trk file bytes and struct view onto bytes"""
    with open(DATA[trk_key], 'rb') as fobj:
        trk_bytes = bytearray(fobj.read())
    dt = trk_module.header_2_dtype.newbyteorder(endian)
    trk_struct = np.ndarray((1,), dt, buffer=trk_bytes)
    trk_struct.flags.writeable = True
    return (trk_struct, trk_bytes)