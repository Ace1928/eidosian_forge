from os.path import basename, isfile, join
from unittest.mock import MagicMock, Mock, patch
import numpy
from numpy import array as npa
from numpy.testing import assert_almost_equal, assert_array_equal
import nibabel
from nibabel.cmdline import parrec2nii
from nibabel.tests.test_parrec import EG_PAR, VARY_PAR
from nibabel.tmpdirs import InTemporaryDirectory
Tests for the parrec2nii exe code
