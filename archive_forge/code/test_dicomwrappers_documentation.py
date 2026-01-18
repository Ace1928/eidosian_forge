import gzip
from copy import copy
from decimal import Decimal
from hashlib import sha1
from os.path import dirname
from os.path import join as pjoin
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...volumeutils import endian_codes
from .. import dicomreaders as didr
from .. import dicomwrappers as didw
from . import dicom_test, have_dicom, pydicom
Make a fake dictionary of data that ``image_shape`` is dependent on.

    Parameters
    ----------
    div_seq : list of tuples
        list of values to use for the `DimensionIndexValues` of each frame.
    sid_seq : list of int
        list of values to use for the `StackID` of each frame.
    sid_dim : int
        the index of the column in 'div_seq' to use as 'sid_seq'
    