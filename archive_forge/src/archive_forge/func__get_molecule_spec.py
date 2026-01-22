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
def _get_molecule_spec(atoms, nuclear_props):
    """ Generate the molecule specification section to write
    to the Gaussian input file, from the Atoms object and dict
    of nuclear properties"""
    molecule_spec = []
    for i, atom in enumerate(atoms):
        symbol_section = atom.symbol + '('
        nuclear_props_set = False
        for keyword, array in nuclear_props.items():
            if array is not None and array[i] is not None:
                string = keyword + '=' + str(array[i]) + ', '
                symbol_section += string
                nuclear_props_set = True
        mass_set = False
        symbol = atom.symbol
        expected_mass = atomic_masses_iupac2016[chemical_symbols.index(symbol)]
        if expected_mass != atoms[i].mass:
            mass_set = True
            string = 'iso' + '=' + str(atoms[i].mass)
            symbol_section += string
        if nuclear_props_set or mass_set:
            symbol_section = symbol_section.strip(', ')
            symbol_section += ')'
        else:
            symbol_section = symbol_section.strip('(')
        molecule_spec.append('{:<10s}{:20.10f}{:20.10f}{:20.10f}'.format(symbol_section, *atom.position))
    for ipbc, tv in zip(atoms.pbc, atoms.cell):
        if ipbc:
            molecule_spec.append('TV {:20.10f}{:20.10f}{:20.10f}'.format(*tv))
    molecule_spec.append('')
    return molecule_spec