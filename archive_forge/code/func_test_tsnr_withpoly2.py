from ...testing import utils
from ..confounds import TSNR
from .. import misc
import pytest
import numpy.testing as npt
from unittest import mock
import nibabel as nb
import numpy as np
import os
def test_tsnr_withpoly2(self):
    tsnrresult = TSNR(in_file=self.in_filenames['in_file'], regress_poly=2).run()
    self.assert_expected_outputs_poly(tsnrresult, {'detrended_file': (-0.22, 8.55), 'mean_file': (2.8, 7.7), 'stddev_file': (0.21, 2.4), 'tsnr_file': (1.7, 35.9)})