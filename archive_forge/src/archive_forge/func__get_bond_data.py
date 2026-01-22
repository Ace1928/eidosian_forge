from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
@staticmethod
def _get_bond_data(line: str, are_multi_center_cobis: bool=False) -> dict:
    """
        Subroutine to extract bond label, site indices, and length from
        a LOBSTER header line. The site indices are zero-based, so they
        can be easily used with a Structure object.

        Example header line: No.4:Fe1->Fe9(2.4524893531900283)
        Example header line for orbital-resolved COHP:
            No.1:Fe1[3p_x]->Fe2[3d_x^2-y^2](2.456180552772262)

        Args:
            line: line in the COHPCAR header describing the bond.
            are_multi_center_cobis: indicates multi-center COBIs

        Returns:
            Dict with the bond label, the bond length, a tuple of the site
            indices, a tuple containing the orbitals (if orbital-resolved),
            and a label for the orbitals (if orbital-resolved).
        """
    if not are_multi_center_cobis:
        line_new = line.rsplit('(', 1)
        length = float(line_new[-1][:-1])
        sites = line_new[0].replace('->', ':').split(':')[1:3]
        site_indices = tuple((int(re.split('\\D+', site)[1]) - 1 for site in sites))
        if '[' in sites[0]:
            orbs = [re.findall('\\[(.*)\\]', site)[0] for site in sites]
            orb_label, orbitals = get_orb_from_str(orbs)
        else:
            orbitals = None
            orb_label = None
        return {'length': length, 'sites': site_indices, 'cells': None, 'orbitals': orbitals, 'orb_label': orb_label}
    line_new = line.rsplit(sep='(', maxsplit=1)
    sites = line_new[0].replace('->', ':').split(':')[1:]
    site_indices = tuple((int(re.split('\\D+', site)[1]) - 1 for site in sites))
    cells = [[int(i) for i in re.split('\\[(.*?)\\]', site)[1].split(' ') if i != ''] for site in sites]
    if sites[0].count('[') > 1:
        orbs = [re.findall('\\]\\[(.*)\\]', site)[0] for site in sites]
        orb_label, orbitals = get_orb_from_str(orbs)
    else:
        orbitals = None
        orb_label = None
    return {'sites': site_indices, 'cells': cells, 'length': None, 'orbitals': orbitals, 'orb_label': orb_label}