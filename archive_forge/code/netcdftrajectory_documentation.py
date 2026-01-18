import os
import warnings
import numpy as np
import ase
from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce
Call pre/post write observers.