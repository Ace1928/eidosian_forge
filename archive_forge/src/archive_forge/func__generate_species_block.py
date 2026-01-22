from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _generate_species_block(self, cond=False):
    """Create a default onetep species block, use -1 for the NGWF number
        to trigger automatic NGWF number assigment using onetep's internal
        routines."""
    if len(self.species) == len(self.atoms.get_chemical_symbols()):
        return
    parameters = self.parameters
    atoms = self.atoms
    if not cond:
        self.species = []
        default_ngwf_radius = self.parameters['ngwf_radius']
        species_ngwf_rad_var = 'species_ngwf_radius'
        species_ngwf_num_var = 'species_ngwf_number'
    else:
        self.species_cond = []
        default_ngwf_radius = self.parameters['ngwf_radius_cond']
        species_ngwf_rad_var = 'species_ngwf_radius_cond'
        species_ngwf_num_var = 'species_ngwf_number_cond'
    for sp in set(zip(atoms.get_atomic_numbers(), atoms.get_chemical_symbols(), ['' if i == 0 else str(i) for i in atoms.get_tags()])):
        try:
            ngrad = parameters[species_ngwf_rad_var][sp[1]]
        except KeyError:
            ngrad = default_ngwf_radius
        try:
            ngnum = parameters[species_ngwf_num_var][sp[1]]
        except KeyError:
            ngnum = -1
        if not cond:
            self.species.append((sp[1] + sp[2], sp[1], sp[0], ngnum, ngrad))
        else:
            self.species_cond.append((sp[1] + sp[2], sp[1], sp[0], ngnum, ngrad))