from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
class Procar:
    """
    Object for reading a PROCAR file.

    Attributes:
        data (dict): The PROCAR data of the form below. It should VASP uses 1-based indexing,
            but all indices are converted to 0-based here.
            { spin: nd.array accessed with (k-point index, band index, ion index, orbital index) }
        weights (np.array): The weights associated with each k-point as an nd.array of length nkpoints.
        phase_factors (dict): Phase factors, where present (e.g. LORBIT = 12). A dict of the form:
            { spin: complex nd.array accessed with (k-point index, band index, ion index, orbital index) }
        nbands (int): Number of bands.
        nkpoints (int): Number of k-points.
        nions (int): Number of ions.
    """

    def __init__(self, filename):
        """
        Args:
            filename: Name of file containing PROCAR.
        """
        headers = None
        with zopen(filename, mode='rt') as file_handle:
            preamble_expr = re.compile('# of k-points:\\s*(\\d+)\\s+# of bands:\\s*(\\d+)\\s+# of ions:\\s*(\\d+)')
            kpoint_expr = re.compile('^k-point\\s+(\\d+).*weight = ([0-9\\.]+)')
            band_expr = re.compile('^band\\s+(\\d+)')
            ion_expr = re.compile('^ion.*')
            expr = re.compile('^([0-9]+)\\s+')
            current_kpoint = 0
            current_band = 0
            done = False
            spin = Spin.down
            weights = None
            for line in file_handle:
                line = line.strip()
                if band_expr.match(line):
                    match = band_expr.match(line)
                    current_band = int(match.group(1)) - 1
                    done = False
                elif kpoint_expr.match(line):
                    match = kpoint_expr.match(line)
                    current_kpoint = int(match.group(1)) - 1
                    weights[current_kpoint] = float(match.group(2))
                    if current_kpoint == 0:
                        spin = Spin.up if spin == Spin.down else Spin.down
                    done = False
                elif headers is None and ion_expr.match(line):
                    headers = line.split()
                    headers.pop(0)
                    headers.pop(-1)
                    data = defaultdict(lambda: np.zeros((n_kpoints, n_bands, n_ions, len(headers))))
                    phase_factors = defaultdict(lambda: np.full((n_kpoints, n_bands, n_ions, len(headers)), np.nan, dtype=np.complex128))
                elif expr.match(line):
                    tokens = line.split()
                    index = int(tokens.pop(0)) - 1
                    num_data = np.array([float(t) for t in tokens[:len(headers)]])
                    if not done:
                        data[spin][current_kpoint, current_band, index, :] = num_data
                    elif len(tokens) > len(headers):
                        num_data = np.array([float(t) for t in tokens[:2 * len(headers)]])
                        for orb in range(len(headers)):
                            phase_factors[spin][current_kpoint, current_band, index, orb] = complex(num_data[2 * orb], num_data[2 * orb + 1])
                    elif np.isnan(phase_factors[spin][current_kpoint, current_band, index, 0]):
                        phase_factors[spin][current_kpoint, current_band, index, :] = num_data
                    else:
                        phase_factors[spin][current_kpoint, current_band, index, :] += 1j * num_data
                elif line.startswith('tot'):
                    done = True
                elif preamble_expr.match(line):
                    match = preamble_expr.match(line)
                    n_kpoints = int(match.group(1))
                    n_bands = int(match.group(2))
                    n_ions = int(match.group(3))
                    weights = np.zeros(n_kpoints)
            self.nkpoints = n_kpoints
            self.nbands = n_bands
            self.nions = n_ions
            self.weights = weights
            self.orbitals = headers
            self.data = data
            self.phase_factors = phase_factors

    def get_projection_on_elements(self, structure: Structure):
        """
        Method returning a dictionary of projections on elements.

        Args:
            structure (Structure): Input structure.

        Returns:
            a dictionary in the {Spin.up:[k index][b index][{Element:values}]]
        """
        dico: dict[Spin, list] = {}
        for spin in self.data:
            dico[spin] = [[defaultdict(float) for i in range(self.nkpoints)] for j in range(self.nbands)]
        for iat in range(self.nions):
            name = structure.species[iat].symbol
            for spin, d in self.data.items():
                for k, b in itertools.product(range(self.nkpoints), range(self.nbands)):
                    dico[spin][b][k][name] += np.sum(d[k, b, iat, :])
        return dico

    def get_occupation(self, atom_index, orbital):
        """
        Returns the occupation for a particular orbital of a particular atom.

        Args:
            atom_num (int): Index of atom in the PROCAR. It should be noted
                that VASP uses 1-based indexing for atoms, but this is
                converted to 0-based indexing in this parser to be
                consistent with representation of structures in pymatgen.
            orbital (str): An orbital. If it is a single character, e.g., s,
                p, d or f, the sum of all s-type, p-type, d-type or f-type
                orbitals occupations are returned respectively. If it is a
                specific orbital, e.g., px, dxy, etc., only the occupation
                of that orbital is returned.

        Returns:
            Sum occupation of orbital of atom.
        """
        orbital_index = self.orbitals.index(orbital)
        return {spin: np.sum(d[:, :, atom_index, orbital_index] * self.weights[:, None]) for spin, d in self.data.items()}