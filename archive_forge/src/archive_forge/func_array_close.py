import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def array_close(val, reference, releps=releps, abseps=abseps):
    valflat = val.flatten()
    refflat = reference.flatten()
    for i, vali in enumerate(valflat):
        close(vali, refflat[i], releps, abseps)