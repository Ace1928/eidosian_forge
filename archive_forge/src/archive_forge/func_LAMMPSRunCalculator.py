import re
import os
from ase.data import atomic_masses, atomic_numbers
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammps import convert
from .kimmodel import KIMModelCalculator
from .exceptions import KIMCalculatorError
def LAMMPSRunCalculator(model_name, model_type, supported_species, options, debug, **kwargs):
    """
    Used for Portable Models or LAMMPS Simulator Models if specifically requested
    """

    def get_params(model_name, supported_units, supported_species, atom_style):
        """
        Extract parameters for LAMMPS calculator from model definition lines.
        Returns a dictionary with entries for "pair_style" and "pair_coeff".
        Expects there to be only one "pair_style" line. There can be multiple
        "pair_coeff" lines (result is returned as a list).
        """
        parameters = {}
        if atom_style:
            parameters['atom_style'] = atom_style
        parameters['units'] = supported_units
        parameters['model_init'] = ['kim_init {} {}{}'.format(model_name, supported_units, os.linesep)]
        parameters['kim_interactions'] = 'kim_interactions {}{}'.format(' '.join(supported_species), os.linesep)
        parameters['masses'] = []
        for i, species in enumerate(supported_species):
            if species not in atomic_numbers:
                raise KIMCalculatorError('Could not determine mass of unknown species {} listed as supported by model'.format(species))
            massstr = str(convert(atomic_masses[atomic_numbers[species]], 'mass', 'ASE', supported_units))
            parameters['masses'].append(str(i + 1) + ' ' + massstr)
        return parameters
    options_not_allowed = ['parameters', 'files', 'specorder', 'keep_tmp_files']
    _check_conflict_options(options, options_not_allowed, simulator='lammpsrun')
    atom_style = kwargs.get('atom_style', None)
    supported_units = kwargs.get('supported_units', 'metal')
    parameters = get_params(model_name, supported_units, supported_species, atom_style)
    return LAMMPS(**parameters, specorder=supported_species, keep_tmp_files=debug, **options)