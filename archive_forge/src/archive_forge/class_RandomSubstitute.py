import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
class RandomSubstitute(Mutation):
    """Substitutes one atom with another atom type. The possible atom types
    are supplied in the parameter elements"""

    def __init__(self, elements=None, num_muts=1, rng=np.random):
        Mutation.__init__(self, num_muts=num_muts, rng=rng)
        self.descriptor = 'RandomSubstitute'
        self.elements = elements

    def substitute(self, atoms):
        """Does the actual substitution"""
        atoms = atoms.copy()
        if self.elements is None:
            elems = list(set(atoms.get_chemical_symbols()))
        else:
            elems = self.elements[:]
        possible_indices = [a.index for a in atoms if a.symbol in elems]
        itbm = self.rng.choice(possible_indices)
        elems.remove(atoms[itbm].symbol)
        new_symbol = self.rng.choice(elems)
        atoms[itbm].symbol = new_symbol
        return atoms

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.substitute(f)
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]
        return (self.finalize_individual(indi), self.descriptor + ':Parent {0}'.format(f.info['confid']))