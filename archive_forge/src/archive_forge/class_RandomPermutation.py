import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
class RandomPermutation(Mutation):
    """Permutes two random atoms.

    Parameters:

    num_muts: the number of times to perform this operation.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, elements=None, num_muts=1, rng=np.random):
        Mutation.__init__(self, num_muts=num_muts, rng=rng)
        self.descriptor = 'RandomPermutation'
        self.elements = elements

    def get_new_individual(self, parents):
        f = parents[0].copy()
        diffatoms = len(set(f.numbers))
        assert diffatoms > 1, 'Permutations with one atomic type is not valid'
        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]
        for _ in range(self.num_muts):
            RandomPermutation.mutate(f, self.elements, rng=self.rng)
        for atom in f:
            indi.append(atom)
        return (self.finalize_individual(indi), self.descriptor + ':Parent {0}'.format(f.info['confid']))

    @classmethod
    def mutate(cls, atoms, elements=None, rng=np.random):
        """Do the actual permutation."""
        if elements is None:
            indices = range(len(atoms))
        else:
            indices = [a.index for a in atoms if a.symbol in elements]
        i1 = rng.choice(indices)
        i2 = rng.choice(indices)
        while atoms[i1].symbol == atoms[i2].symbol:
            i2 = rng.choice(indices)
        atoms.symbols[[i1, i2]] = atoms.symbols[[i2, i1]]