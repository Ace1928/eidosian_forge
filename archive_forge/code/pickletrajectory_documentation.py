import os
import sys
import errno
import pickle
import warnings
import collections
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError
from ase.constraints import FixAtoms
from ase.parallel import world, barrier
Call pre/post write observers.