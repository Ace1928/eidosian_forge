import pytest
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from .test_ce_curvature import Al_atom_pair
This test ensures that the potentiostat is working even when curvature
    extrapolation (use_fs) is turned off.