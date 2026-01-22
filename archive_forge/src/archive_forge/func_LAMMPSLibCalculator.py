import re
import os
from ase.data import atomic_masses, atomic_numbers
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammps import convert
from .kimmodel import KIMModelCalculator
from .exceptions import KIMCalculatorError
def LAMMPSLibCalculator(model_name, supported_species, supported_units, options):
    """
    Only used for LAMMPS Simulator Models
    """
    options_not_allowed = ['lammps_header', 'lmpcmds', 'atom_types', 'log_file', 'keep_alive']
    _check_conflict_options(options, options_not_allowed, simulator='lammpslib')
    model_init = ['units ' + supported_units + os.linesep]
    model_init.append('kim_init {} {}{}'.format(model_name, supported_units, os.linesep))
    model_init.append('atom_modify map array sort 0 0' + os.linesep)
    atom_types = {}
    for i_s, s in enumerate(supported_species):
        atom_types[s] = i_s + 1
    kim_interactions = ['kim_interactions {}'.format(' '.join(supported_species))]
    return LAMMPSlib(lammps_header=model_init, lammps_name=None, lmpcmds=kim_interactions, post_changebox_cmds=kim_interactions, atom_types=atom_types, log_file='lammps.log', keep_alive=True, **options)