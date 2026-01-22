import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
class Rich2poorPermutation(_NeighborhoodPermutation):
    """
    The rich to poor (Rich2poor) permutation operator described in
    S. Lysgaard et al., Top. Catal., 2014, 57 (1-4), pp 33-39

    Permutes two atoms from regions rich in the same elements, to
    regions short of the same elements.
    (Inverse of Poor2richPermutation)

    Parameters:

    elements: Which elements to take into account in this permutation

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, elements=None, num_muts=1, rng=np.random):
        _NeighborhoodPermutation.__init__(self, num_muts=num_muts, rng=rng)
        self.descriptor = 'Rich2poorPermutation'
        self.elements = elements

    def get_new_individual(self, parents):
        f = parents[0].copy()
        diffatoms = len(set(f.numbers))
        assert diffatoms > 1, 'Permutations with one atomic type is not valid'
        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]
        if self.elements is None:
            elems = list(set(f.get_chemical_symbols()))
        else:
            elems = self.elements
        for _ in range(self.num_muts):
            Rich2poorPermutation.mutate(f, elems, rng=self.rng)
        for atom in f:
            indi.append(atom)
        return (self.finalize_individual(indi), self.descriptor + ':Parent {0}'.format(f.info['confid']))

    @classmethod
    def mutate(cls, atoms, elements, rng=np.random):
        _NP = _NeighborhoodPermutation
        ac = atoms.copy()
        del ac[[atom.index for atom in ac if atom.symbol not in elements]]
        permuts = _NP.get_possible_poor2rich_permutations(ac, inverse=True)
        swap = list(rng.choice(permuts))
        atoms.symbols[swap] = atoms.symbols[swap[::-1]]