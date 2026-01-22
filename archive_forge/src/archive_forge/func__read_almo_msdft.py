from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def _read_almo_msdft(self):
    """Parse output of ALMO(MSDFT) calculations for coupling between diabatic states."""
    temp_dict = read_pattern(self.text, {'states': 'Number of diabatic states: 2\\s*\\nstate 1\\s*\\ncharge per fragment\\s*\\n((?:\\s*[\\-0-9]+\\s*\\n)+)multiplicity per fragment\\s*\\n((?:\\s*[\\-0-9]+\\s*\\n)+)state 2\\s*\\ncharge per fragment\\s*\\n((?:\\s*[\\-0-9]+\\s*\\n)+)multiplicity per fragment\\s*\\n((?:\\s*[\\-0-9]+\\s*\\n)+)', 'diabat_energies': 'Energies of the diabats:\\s*\\n\\s*state 1:\\s+([\\-\\.0-9]+)\\s*\\n\\s*state 2:\\s+([\\-\\.0-9]+)', 'adiabat_energies': 'Energy of the adiabatic states\\s*\\n\\s*State 1:\\s+([\\-\\.0-9]+)\\s*\\n\\s*State 2:\\s+([\\-\\.0-9]+)', 'hamiltonian': 'Hamiltonian\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 'overlap': 'overlap\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 's2': '<S2>\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 'diabat_basis_coeff': 'Diabatic basis coefficients\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 'h_coupling': 'H passed to diabatic coupling calculation\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 'coupling': 'Coupling between diabats 1 and 2: (?:[\\-\\.0-9]+) \\(([\\-\\.0-9]+) meV\\)'})
    if temp_dict.get('states') is None or len(temp_dict.get('states', [])) == 0:
        self.data['almo_coupling_states'] = None
    else:
        charges_1 = [int(r.strip()) for r in temp_dict['states'][0][0].strip().split('\n')]
        spins_1 = [int(r.strip()) for r in temp_dict['states'][0][1].strip().split('\n')]
        charges_2 = [int(r.strip()) for r in temp_dict['states'][0][2].strip().split('\n')]
        spins_2 = [int(r.strip()) for r in temp_dict['states'][0][3].strip().split('\n')]
        self.data['almo_coupling_states'] = [[[i, j] for i, j in zip(charges_1, spins_1)], [[i, j] for i, j in zip(charges_2, spins_2)]]
    if temp_dict.get('diabat_energies') is None or len(temp_dict.get('diabat_energies', [])) == 0:
        self.data['almo_diabat_energies_Hartree'] = None
    else:
        self.data['almo_diabat_energies_Hartree'] = [float(x) for x in temp_dict['diabat_energies'][0]]
    if temp_dict.get('adiabat_energies') is None or len(temp_dict.get('adiabat_energies', [])) == 0:
        self.data['almo_adiabat_energies_Hartree'] = None
    else:
        self.data['almo_adiabat_energies_Hartree'] = [float(x) for x in temp_dict['adiabat_energies'][0]]
    if temp_dict.get('hamiltonian') is None or len(temp_dict.get('hamiltonian', [])) == 0:
        self.data['almo_hamiltonian'] = None
    else:
        self.data['almo_hamiltonian'] = [[float(temp_dict['hamiltonian'][0][0]), float(temp_dict['hamiltonian'][0][1])], [float(temp_dict['hamiltonian'][0][2]), float(temp_dict['hamiltonian'][0][3])]]
    if temp_dict.get('overlap') is None or len(temp_dict.get('overlap', [])) == 0:
        self.data['almo_overlap_matrix'] = None
    else:
        self.data['almo_overlap_matrix'] = [[float(temp_dict['overlap'][0][0]), float(temp_dict['overlap'][0][1])], [float(temp_dict['overlap'][0][2]), float(temp_dict['overlap'][0][3])]]
    if temp_dict.get('s2') is None or len(temp_dict.get('s2', [])) == 0:
        self.data['almo_s2_matrix'] = None
    else:
        self.data['almo_s2_matrix'] = [[float(temp_dict['s2'][0][0]), float(temp_dict['s2'][0][1])], [float(temp_dict['s2'][0][2]), float(temp_dict['s2'][0][3])]]
    if temp_dict.get('diabat_basis_coeff') is None or len(temp_dict.get('diabat_basis_coeff', [])) == 0:
        self.data['almo_diabat_basis_coeff'] = None
    else:
        self.data['almo_diabat_basis_coeff'] = [[float(temp_dict['diabat_basis_coeff'][0][0]), float(temp_dict['diabat_basis_coeff'][0][1])], [float(temp_dict['diabat_basis_coeff'][0][2]), float(temp_dict['diabat_basis_coeff'][0][3])]]
    if temp_dict.get('h_coupling') is None or len(temp_dict.get('h_coupling', [])) == 0:
        self.data['almo_h_coupling_matrix'] = None
    else:
        self.data['almo_h_coupling_matrix'] = [[float(temp_dict['h_coupling'][0][0]), float(temp_dict['h_coupling'][0][1])], [float(temp_dict['h_coupling'][0][2]), float(temp_dict['h_coupling'][0][3])]]
    if temp_dict.get('coupling') is None or len(temp_dict.get('coupling', [])) == 0:
        self.data['almo_coupling_eV'] = None
    else:
        self.data['almo_coupling_eV'] = float(temp_dict['coupling'][0][0]) / 1000