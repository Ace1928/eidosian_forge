import numpy as np
import pytest
from ase import Atoms
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.bond_polarizability import LippincottStuttman, Linearized
Compare polarizabilties of one and two bonds