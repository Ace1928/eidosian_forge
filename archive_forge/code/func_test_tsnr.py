from ...testing import utils
from ..confounds import TSNR
from .. import misc
import pytest
import numpy.testing as npt
from unittest import mock
import nibabel as nb
import numpy as np
import os
def test_tsnr(self):
    tsnrresult = TSNR(in_file=self.in_filenames['in_file']).run()
    self.assert_expected_outputs(tsnrresult, {'mean_file': (2.8, 7.4), 'stddev_file': (0.8, 2.9), 'tsnr_file': (1.3, 9.25)})