import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.calculators.calculator import InputError, Calculator
from ase.calculators.gaussian import Gaussian
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses_iupac2016, chemical_symbols
from ase.io import ParseError
from ase.io.zmatrix import parse_zmatrix
from ase.units import Bohr, Hartree
def _get_atoms_from_molspec(molspec_section):
    """ Takes a string: molspec_section which contains the molecule
    specification section of a gaussian input file, and returns an atoms
    object that represents this."""
    symbols = []
    positions = []
    pbc = np.zeros(3, dtype=bool)
    cell = np.zeros((3, 3))
    npbc = 0
    nuclear_props = []
    zmatrix_type = False
    zmatrix_contents = ''
    zmatrix_var_section = False
    zmatrix_vars = ''
    for line in molspec_section:
        line = line.split('!')[0].replace('/', ' ').replace(',', ' ')
        if line.split():
            if zmatrix_type:
                if zmatrix_var_section:
                    zmatrix_vars += line.strip() + '\n'
                    continue
                elif 'variables' in line.lower():
                    zmatrix_var_section = True
                    continue
                elif 'constants' in line.lower():
                    zmatrix_var_section = True
                    warnings.warn('Constants in the optimisation are not currently supported. Instead setting constants as variables.')
                    continue
            symbol, pos = _get_atoms_info(line)
            current_nuclear_props = _get_nuclear_props(line)
            if not zmatrix_type:
                pos = _get_cartesian_atom_coords(symbol, pos)
                if pos is None:
                    zmatrix_type = True
                if symbol.upper() == 'TV' and pos is not None:
                    pbc[npbc] = True
                    cell[npbc] = pos
                    npbc += 1
                else:
                    nuclear_props.append(current_nuclear_props)
                    if not zmatrix_type:
                        symbols.append(symbol)
                        positions.append(pos)
            if zmatrix_type:
                zmatrix_contents += _get_zmatrix_line(line)
    if len(positions) == 0:
        if zmatrix_type:
            if zmatrix_vars == '':
                zmatrix_vars = None
            positions, symbols = _read_zmatrix(zmatrix_contents, zmatrix_vars)
    try:
        atoms = Atoms(symbols, positions, pbc=pbc, cell=cell)
    except (IndexError, ValueError, KeyError) as e:
        raise ParseError('ERROR: Could not read the Gaussian input file, due to a problem with the molecule specification: {}'.format(e))
    nuclear_props = _get_nuclear_props_for_all_atoms(nuclear_props)
    return (atoms, nuclear_props)