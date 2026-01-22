import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
class RandomMutation(Mutation):
    """Moves a random atom the supplied length in a random direction."""

    def __init__(self, length=2.0, num_muts=1, rng=np.random):
        Mutation.__init__(self, num_muts=num_muts, rng=rng)
        self.descriptor = 'RandomMutation'
        self.length = length

    def mutate(self, atoms):
        """ Does the actual mutation. """
        tbm = self.rng.choice(range(len(atoms)))
        indi = Atoms()
        for a in atoms:
            if a.index == tbm:
                a.position += self.random_vector(self.length, rng=self.rng)
            indi.append(a)
        return indi

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]
        to_mut = f.copy()
        for _ in range(self.num_muts):
            to_mut = self.mutate(to_mut)
        for atom in to_mut:
            indi.append(atom)
        return (self.finalize_individual(indi), self.descriptor + ':Parent {0}'.format(f.info['confid']))

    @classmethod
    def random_vector(cls, l, rng=np.random):
        """return random vector of length l"""
        vec = np.array([rng.rand() * 2 - 1 for i in range(3)])
        vl = np.linalg.norm(vec)
        return np.array([v * l / vl for v in vec])