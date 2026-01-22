from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from tabulate import tabulate
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.core import ParseError
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
class Potential(MSONable):
    """FEFF atomic potential."""

    def __init__(self, struct, absorbing_atom):
        """
        Args:
            struct (Structure): Structure object.
            absorbing_atom (str/int): Absorbing atom symbol or site index.
        """
        if struct.is_ordered:
            self.struct = struct
            atom_sym = get_absorbing_atom_symbol_index(absorbing_atom, struct)[0]
            self.pot_dict = get_atom_map(struct, atom_sym)
        else:
            raise ValueError('Structure with partial occupancies cannot be converted into atomic coordinates!')
        self.absorbing_atom, _ = get_absorbing_atom_symbol_index(absorbing_atom, struct)

    @staticmethod
    def pot_string_from_file(filename='feff.inp'):
        """
        Reads Potential parameters from a feff.inp or FEFFPOT file.
        The lines are arranged as follows:

          ipot   Z   element   lmax1   lmax2   stoichometry   spinph

        Args:
            filename: file name containing potential data.

        Returns:
            FEFFPOT string.
        """
        with zopen(filename, mode='rt') as f_object:
            f = f_object.readlines()
            ln = -1
            pot_str = ['POTENTIALS\n']
            pot_tag = -1
            pot_data = 0
            pot_data_over = 1
            sep_line_pattern = [re.compile('ipot.*Z.*tag.*lmax1.*lmax2.*spinph'), re.compile('^[*]+.*[*]+$')]
            for line in f:
                if pot_data_over == 1:
                    ln += 1
                    if pot_tag == -1:
                        pot_tag = line.find('POTENTIALS')
                        ln = 0
                    if pot_tag >= 0 and ln > 0 and (pot_data_over > 0):
                        try:
                            if len(sep_line_pattern[0].findall(line)) > 0 or len(sep_line_pattern[1].findall(line)) > 0:
                                pot_str.append(line)
                            elif int(line.split()[0]) == pot_data:
                                pot_data += 1
                                pot_str.append(line.replace('\r', ''))
                        except (ValueError, IndexError):
                            if pot_data > 0:
                                pot_data_over = 0
        return ''.join(pot_str).rstrip('\n')

    @staticmethod
    def pot_dict_from_str(pot_data):
        """
        Creates atomic symbol/potential number dictionary
        forward and reverse.

        Args:
            pot_data: potential data in string format

        Returns:
            forward and reverse atom symbol and potential number dictionaries.
        """
        pot_dict = {}
        pot_dict_reverse = {}
        begin = 0
        ln = -1
        for line in pot_data.split('\n'):
            try:
                if begin == 0 and line.split()[0] == '0':
                    begin += 1
                    ln = 0
                if begin == 1:
                    ln += 1
                if ln > 0:
                    atom = line.split()[2]
                    index = int(line.split()[0])
                    pot_dict[atom] = index
                    pot_dict_reverse[index] = atom
            except (ValueError, IndexError):
                pass
        return (pot_dict, pot_dict_reverse)

    def __str__(self):
        """
        Returns a string representation of potential parameters to be used in
        the feff.inp file,
        determined from structure object.

                The lines are arranged as follows:

            ipot   Z   element   lmax1   lmax2   stoichiometry   spinph

        Returns:
            String representation of Atomic Coordinate Shells.
        """
        central_element = Element(self.absorbing_atom)
        ipotrow = [[0, central_element.Z, central_element.symbol, -1, -1, 0.0001, 0]]
        for el, amt in self.struct.composition.items():
            if el == central_element and amt == 1:
                continue
            ipot = self.pot_dict[el.symbol]
            ipotrow.append([ipot, el.Z, el.symbol, -1, -1, amt, 0])
        ipot_sorted = sorted(ipotrow, key=lambda x: x[0])
        ipotrow = str(tabulate(ipot_sorted, headers=['*ipot', 'Z', 'tag', 'lmax1', 'lmax2', 'xnatph(stoichometry)', 'spinph']))
        ipotlist = ipotrow.replace('--', '**')
        return f'POTENTIALS \n{ipotlist}'

    def write_file(self, filename='POTENTIALS'):
        """
        Write to file.

        Args:
            filename: filename and path to write potential file to.
        """
        with zopen(filename, mode='wt') as file:
            file.write(str(self) + '\n')