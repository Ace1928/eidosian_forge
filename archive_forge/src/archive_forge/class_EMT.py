import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT as OrigEMT
from ase.dyneb import DyNEB
from ase.optimize import BFGS
class EMT(OrigEMT):

    def calculate(self, *args, **kwargs):
        force_evaluations[0] += 1
        OrigEMT.calculate(self, *args, **kwargs)