import numpy as np
import numpy.testing as npt
from .. import rapidart as ra
from ...interfaces.base import Bunch
def test_sc_output_filenames():
    sc = ra.StimulusCorrelation()
    outputdir = '/tmp'
    f = 'motion.nii'
    corrfile = sc._get_output_filenames(f, outputdir)
    assert corrfile == '/tmp/qa.motion_stimcorr.txt'