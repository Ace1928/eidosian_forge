import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.test import FreeElectrons
from ase.dft.kpoints import special_paths
from ase.spectrum.band_structure import BandStructure
def test_print_bs(bs):
    print(bs)