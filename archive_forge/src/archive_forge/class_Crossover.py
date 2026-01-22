from ase.ga.offspring_creator import OffspringCreator
from ase import Atoms
from itertools import chain
import numpy as np
class Crossover(OffspringCreator):
    """Base class for all particle crossovers.

    Originally intended for medium sized particles

    Do not call this class directly."""

    def __init__(self, rng=np.random):
        OffspringCreator.__init__(self, rng=rng)
        self.descriptor = 'Crossover'
        self.min_inputs = 2