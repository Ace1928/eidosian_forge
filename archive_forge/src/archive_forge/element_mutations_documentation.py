import numpy as np
from ase.data import atomic_numbers
from ase.ga.offspring_creator import OffspringCreator
Returns a list of the indices that are going to
        be mutated and a list of possible elements to mutate
        to. The lists obey the criteria set in the initialization.
        