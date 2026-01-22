from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
class RandomSlabPermutation(SlabOperator):

    def __init__(self, verbose=False, num_muts=1, allowed_compositions=None, distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts, allowed_compositions, distribution_correction_function, rng=rng)
        self.descriptor = 'RandomSlabPermutation'

    def get_new_individual(self, parents):
        f = parents[0]
        if len(set(f.get_chemical_symbols())) == 1:
            f = parents[1]
            if len(set(f.get_chemical_symbols())) == 1:
                return (None, '{1} not possible in {0}'.format(f.info['confid'], self.descriptor))
        indi = self.initialize_individual(f, f)
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]
        indi = self.operate(indi)
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi), self.descriptor + parent_message)

    def operate(self, atoms):
        for _ in range(self.num_muts):
            permute2(atoms, rng=self.rng)
        self.dcf(atoms)
        return atoms