import pytest
from ase import Atoms
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
This test has two atoms spinning counter-clockwise around eachother. the
    The radius (1/curvature) is less obviously pair_distance*sqrt(2)/2.
    This is the simplest multi-body analytic curvature test.