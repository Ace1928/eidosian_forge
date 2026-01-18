from io import StringIO
from os.path import join as pjoin
import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline.diff import *
from nibabel.cmdline.utils import *
from nibabel.testing import data_path
def test_table2string():
    assert table2string([['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H']]) == 'A B C D\nE F G H\n'
    assert table2string([["Let's", 'Make', 'Tests', 'And'], ['Have', 'Lots', 'Of', 'Fun'], ['With', 'Python', 'Guys', '!']]) == "Let's  Make  Tests And\n Have  Lots    Of  Fun" + '\n With Python  Guys  !\n'