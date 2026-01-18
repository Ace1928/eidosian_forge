import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_tcompcor(self):
    ccinterface = TCompCor(num_components=6, realigned_file=self.realigned_file, percentile_threshold=0.75)
    self.run_cc(ccinterface, [[-0.111453619, -0.4632908609], [0.456690731, 0.6983205193], [-0.7132557407, 0.1340170559], [0.5022537643, -0.5098322262], [-0.1342351356, 0.1407855119]], 'tCompCor')