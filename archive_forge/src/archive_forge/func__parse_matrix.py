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
def _parse_matrix(file_data, pattern, e_fermi):
    complex_matrices = {}
    matrix_diagonal_values = []
    start_inxs_real = []
    end_inxs_real = []
    start_inxs_imag = []
    end_inxs_imag = []
    for idx, line in enumerate(file_data):
        line = line.strip()
        if 'Real parts' in line:
            start_inxs_real += [idx + 1]
            if idx == 1:
                pass
            else:
                end_inxs_imag += [idx - 1]
            matches = re.search(pattern, file_data[idx - 1])
            if matches and len(matches.groups()) == 2:
                k_point = matches.group(2)
                complex_matrices[k_point] = {}
        if 'Imag parts' in line:
            end_inxs_real += [idx - 1]
            start_inxs_imag += [idx + 1]
        if idx == len(file_data) - 1:
            end_inxs_imag += [len(file_data)]
    matrix_real = []
    matrix_imag = []
    for start_inx_real, end_inx_real, start_inx_imag, end_inx_imag in zip(start_inxs_real, end_inxs_real, start_inxs_imag, end_inxs_imag):
        matrix_real = file_data[start_inx_real:end_inx_real]
        matrix_imag = file_data[start_inx_imag:end_inx_imag]
        matrix_array_real = np.array([line.split()[1:] for line in matrix_real[1:]], dtype=float)
        matrix_array_imag = np.array([line.split()[1:] for line in matrix_imag[1:]], dtype=float)
        comp_matrix = matrix_array_real + 1j * matrix_array_imag
        matches = re.search(pattern, file_data[start_inx_real - 2])
        if matches and len(matches.groups()) == 2:
            spin = Spin.up if matches.group(1) == '1' else Spin.down
            k_point = matches.group(2)
            complex_matrices[k_point].update({spin: comp_matrix})
        elif matches and len(matches.groups()) == 1:
            k_point = matches.group(1)
            complex_matrices.update({k_point: comp_matrix})
        matrix_diagonal_values += [comp_matrix.real.diagonal() - e_fermi]
    elements_basis_functions = [line.split()[:1][0] for line in matrix_real if line.split()[:1][0] != 'basisfunction']
    average_matrix_diagonal_values = np.array(matrix_diagonal_values, dtype=float).mean(axis=0)
    average_average_matrix_diag_dict = dict(zip(elements_basis_functions, average_matrix_diagonal_values))
    return (matrix_diagonal_values, average_average_matrix_diag_dict, complex_matrices)