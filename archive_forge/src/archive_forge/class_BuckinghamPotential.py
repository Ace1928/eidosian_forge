from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class BuckinghamPotential:
    """Generate the Buckingham Potential Table from the bush.lib and lewis.lib.

    Ref:
    T.S.Bush, J.D.Gale, C.R.A.Catlow and P.D. Battle,  J. Mater Chem.,
    4, 831-837 (1994).
    G.V. Lewis and C.R.A. Catlow, J. Phys. C: Solid State Phys., 18,
    1149-1161 (1985)
    """

    def __init__(self, bush_lewis_flag):
        """
        Args:
            bush_lewis_flag (str): Flag for using Bush or Lewis potential.
        """
        assert bush_lewis_flag in {'bush', 'lewis'}
        pot_file = 'bush.lib' if bush_lewis_flag == 'bush' else 'lewis.lib'
        with open(os.path.join(os.environ['GULP_LIB'], pot_file)) as file:
            species_dict, pot_dict, spring_dict = ({}, {}, {})
            sp_flg, pot_flg, spring_flg = (False, False, False)
            for row in file:
                if row[0] == '#':
                    continue
                if row.split()[0] == 'species':
                    sp_flg, pot_flg, spring_flg = (True, False, False)
                    continue
                if row.split()[0] == 'buckingham':
                    sp_flg, pot_flg, spring_flg = (False, True, False)
                    continue
                if row.split()[0] == 'spring':
                    sp_flg, pot_flg, spring_flg = (False, False, True)
                    continue
                elmnt = row.split()[0]
                if sp_flg:
                    if bush_lewis_flag == 'bush':
                        if elmnt not in species_dict:
                            species_dict[elmnt] = {'inp_str': '', 'oxi': 0}
                        species_dict[elmnt]['inp_str'] += row
                        species_dict[elmnt]['oxi'] += float(row.split()[2])
                    elif bush_lewis_flag == 'lewis':
                        if elmnt == 'O':
                            if row.split()[1] == 'core':
                                species_dict['O_core'] = row
                            if row.split()[1] == 'shel':
                                species_dict['O_shel'] = row
                        else:
                            metal = elmnt.split('_')[0]
                            species_dict[elmnt] = f'{metal} core {row.split()[2]}\n'
                    continue
                if pot_flg:
                    if bush_lewis_flag == 'bush':
                        pot_dict[elmnt] = row
                    elif bush_lewis_flag == 'lewis':
                        if elmnt == 'O':
                            pot_dict['O'] = row
                        else:
                            metal = elmnt.split('_')[0]
                            pot_dict[elmnt] = f'{metal} {' '.join(row.split()[1:])}\n'
                    continue
                if spring_flg:
                    spring_dict[elmnt] = row
            if bush_lewis_flag == 'bush':
                for key in pot_dict:
                    if key not in spring_dict:
                        spring_dict[key] = ''
            self.species_dict = species_dict
            self.pot_dict = pot_dict
            self.spring_dict = spring_dict