from pathlib import Path
import numpy as np
import pytest
from ase.utils.xrdebye import XrDebye, wavelengths
from ase.cluster.cubic import FaceCenteredCubic
def test_xrd(testdir, xrd):
    expected = np.array([18549.274677, 52303.116995, 38502.372027])
    obtained = xrd.calc_pattern(x=np.array([15, 30, 50]), mode='XRD')
    assert np.allclose(obtained, expected, rtol=tolerance)
    xrd.write_pattern('tmp.txt')
    assert Path('tmp.txt').exists()