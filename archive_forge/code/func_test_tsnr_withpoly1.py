from ...testing import utils
from ..confounds import TSNR
from .. import misc
import pytest
import numpy.testing as npt
from unittest import mock
import nibabel as nb
import numpy as np
import os
def test_tsnr_withpoly1(self):
    tsnrresult = TSNR(in_file=self.in_filenames['in_file'], regress_poly=1).run()
    self.assert_expected_outputs_poly(tsnrresult, {'detrended_file': (-0.1, 8.7), 'mean_file': (2.8, 7.4), 'stddev_file': (0.75, 2.75), 'tsnr_file': (1.4, 9.9)})