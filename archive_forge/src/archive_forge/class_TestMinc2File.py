from os.path import join as pjoin
import numpy as np
import pytest
from .. import minc2
from ..minc2 import Minc2File, Minc2Image
from ..optpkg import optional_package
from ..testing import data_path
from . import test_minc1 as tm2
class TestMinc2File(tm2._TestMincFile):
    module = minc2
    file_class = Minc2File
    opener = h5py.File
    test_files = EXAMPLE_IMAGES