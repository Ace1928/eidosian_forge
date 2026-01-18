import numpy as np
from ase.build import bulk
from ase.io.aims import read_aims as read
from ase.io.aims import parse_geometry_lines
from pytest import approx
write fractional coords and check if structure was preserved