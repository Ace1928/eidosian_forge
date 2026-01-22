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
def _read_geometries(self):
    """Parses all geometries from an optimization trajectory."""
    geoms = []
    if self.data.get('new_optimizer') is None:
        header_pattern = '\\s+Optimization\\sCycle:\\s+\\d+\\s+Coordinates \\(Angstroms\\)\\s+ATOM\\s+X\\s+Y\\s+Z'
        table_pattern = '\\s+\\d+\\s+\\w+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)'
        footer_pattern = '\\s+Point Group\\:\\s+[\\d\\w\\*]+\\s+Number of degrees of freedom\\:\\s+\\d+'
    elif read_pattern(self.text, {'key': 'Geometry Optimization Coordinates :\\s+Cartesian'}, terminate_on_match=True).get('key') == [[]]:
        header_pattern = 'RMS\\s+of Stepsize\\s+[\\d\\-\\.]+\\s+-+\\s+Standard Nuclear Orientation \\(Angstroms\\)\\s+I\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
        table_pattern = '\\s*\\d+\\s+[a-zA-Z]+\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*'
        footer_pattern = '\\s*-+'
    else:
        header_pattern = 'Finished Iterative Coordinate Back-Transformation\\s+-+\\s+Standard Nuclear Orientation \\(Angstroms\\)\\s+I\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
        table_pattern = '\\s*\\d+\\s+[a-zA-Z]+\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*'
        footer_pattern = '\\s*-+'
    parsed_geometries = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
    for parsed_geometry in parsed_geometries:
        if not parsed_geometry:
            geoms.append(None)
        else:
            geoms.append(process_parsed_coords(parsed_geometry))
    if len(geoms) >= 1:
        self.data['geometries'] = geoms
        self.data['last_geometry'] = geoms[-1]
        if self.data.get('charge') is not None:
            self.data['molecule_from_last_geometry'] = Molecule(species=self.data.get('species'), coords=self.data.get('last_geometry'), charge=self.data.get('charge'), spin_multiplicity=self.data.get('multiplicity'))
        if self.data.get('new_optimizer') is None:
            header_pattern = '\\*+\\s+(OPTIMIZATION|TRANSITION STATE)\\s+CONVERGED\\s+\\*+\\s+\\*+\\s+Coordinates \\(Angstroms\\)\\s+ATOM\\s+X\\s+Y\\s+Z'
            table_pattern = '\\s+\\d+\\s+\\w+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)'
            footer_pattern = '\\s+Z-matrix Print:'
        else:
            header_pattern = '(OPTIMIZATION|TRANSITION STATE)\\sCONVERGED\\s+\\*+\\s+\\*+\\s+-+\\s+Standard Nuclear Orientation \\(Angstroms\\)\\s+I\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
            table_pattern = '\\s*\\d+\\s+[a-zA-Z]+\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*'
            footer_pattern = '\\s*-+'
        parsed_optimized_geometries = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
        if not parsed_optimized_geometries:
            self.data['optimized_geometry'] = None
            header_pattern = '^\\s+\\*+\\s+(OPTIMIZATION|TRANSITION STATE) CONVERGED\\s+\\*+\\s+\\*+\\s+Z-matrix\\s+Print:\\s+\\$molecule\\s+[\\d\\-]+\\s+[\\d\\-]+\\n'
            table_pattern = '\\s*(\\w+)(?:\\s+(\\d+)\\s+([\\d\\-\\.]+)(?:\\s+(\\d+)\\s+([\\d\\-\\.]+)(?:\\s+(\\d+)\\s+([\\d\\-\\.]+))*)*)*(?:\\s+0)*'
            footer_pattern = '^\\$end\\n'
            self.data['optimized_zmat'] = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
        else:
            self.data['optimized_geometry'] = process_parsed_coords(parsed_optimized_geometries[0])
            self.data['optimized_geometries'] = [process_parsed_coords(i) for i in parsed_optimized_geometries]
            if self.data.get('charge') is not None:
                self.data['molecule_from_optimized_geometry'] = Molecule(species=self.data.get('species'), coords=self.data.get('optimized_geometry'), charge=self.data.get('charge'), spin_multiplicity=self.data.get('multiplicity'))
                self.data['molecules_from_optimized_geometries'] = []
                for geom in self.data['optimized_geometries']:
                    mol = Molecule(species=self.data.get('species'), coords=geom, charge=self.data.get('charge'), spin_multiplicity=self.data.get('multiplicity'))
                    self.data['molecules_from_optimized_geometries'].append(mol)