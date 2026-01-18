import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.test import FreeElectrons
from ase.dft.kpoints import special_paths
from ase.spectrum.band_structure import BandStructure
def test_subtract_ref(bs):
    avg = np.mean(bs.energies)
    bs._reference = 5
    bs2 = bs.subtract_reference()
    avg2 = np.mean(bs2.energies)
    assert avg - 5 == pytest.approx(avg2)