import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..analyze import AnalyzeImage
from ..funcs import OrientationError, as_closest_canonical, concat_images
from ..loadsave import save
from ..nifti1 import Nifti1Image
from ..tmpdirs import InTemporaryDirectory
Test for image funcs