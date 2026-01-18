import time
from math import sqrt
from os.path import isfile
from ase.io.jsonio import read_json, write_json
from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import world, barrier
from ase.io.trajectory import Trajectory
from ase.utils import IOContext
import collections.abc
Automatically sets force_consistent to True if force_consistent
        energies are supported by calculator; else False.