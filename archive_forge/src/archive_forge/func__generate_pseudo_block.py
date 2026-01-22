from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _generate_pseudo_block(self):
    """Create a default onetep pseudopotentials block, using the
        element name with the variable pseudo_suffix appended to it by default,
        unless the user has set overrides for specific species by setting
        specific entries in species_pseudo"""
    for sp in self.species:
        try:
            pseudo_string = self.parameters['species_pseudo'][sp[0]]
        except KeyError:
            try:
                pseudo_string = sp[1] + self.parameters['pseudo_suffix']
            except KeyError:
                pseudo_string = sp[1]
        self.pseudos.append((sp[0], pseudo_string))