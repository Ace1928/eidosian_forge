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
def _read_gradients(self):
    """Parses all gradients obtained during an optimization trajectory."""
    grad_header_pattern = 'Gradient of (?:SCF)?(?:MP2)? Energy(?: \\(in au\\.\\))?'
    footer_pattern = '(?:Max gradient component|Gradient time)'
    grad_format_length = self._get_grad_format_length(grad_header_pattern)
    grad_table_pattern = '(?:\\s+\\d+(?:\\s+\\d+)?(?:\\s+\\d+)?(?:\\s+\\d+)?(?:\\s+\\d+)?(?:\\s+\\d+)?)?\\n\\s\\s\\s\\s[1-3]\\s*(\\-?[\\d\\.]{9,12})'
    if grad_format_length > 1:
        for _ in range(1, grad_format_length):
            grad_table_pattern = grad_table_pattern + '(?:\\s*(\\-?[\\d\\.]{9,12}))?'
    parsed_gradients = read_table_pattern(self.text, grad_header_pattern, grad_table_pattern, footer_pattern)
    if len(parsed_gradients) >= 1:
        sorted_gradients = np.zeros(shape=(len(parsed_gradients), len(self.data['initial_molecule']), 3))
        for ii, grad in enumerate(parsed_gradients):
            for jj in range(int(len(grad) / 3)):
                for kk in range(grad_format_length):
                    if grad[jj * 3][kk] != 'None':
                        sorted_gradients[ii][jj * grad_format_length + kk][0] = grad[jj * 3][kk]
                        sorted_gradients[ii][jj * grad_format_length + kk][1] = grad[jj * 3 + 1][kk]
                        sorted_gradients[ii][jj * grad_format_length + kk][2] = grad[jj * 3 + 2][kk]
        self.data['gradients'] = sorted_gradients
        if self.data['solvent_method'] is not None:
            header_pattern = 'total gradient after adding PCM contribution --\\s+-+\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
            table_pattern = '\\s+\\d+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s'
            footer_pattern = '-+'
            parsed_gradients = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
            pcm_gradients = np.zeros(shape=(len(parsed_gradients), len(self.data['initial_molecule']), 3))
            for ii, grad in enumerate(parsed_gradients):
                for jj, entry in enumerate(grad):
                    for kk, val in enumerate(entry):
                        pcm_gradients[ii][jj][kk] = float(val)
            self.data['pcm_gradients'] = pcm_gradients
        else:
            self.data['pcm_gradients'] = None
        if read_pattern(self.text, {'key': 'Gradient of CDS energy'}, terminate_on_match=True).get('key') == [[]]:
            header_pattern = 'Gradient of CDS energy'
            parsed_gradients = read_table_pattern(self.text, header_pattern, grad_table_pattern, grad_header_pattern)
            sorted_gradients = np.zeros(shape=(len(parsed_gradients), len(self.data['initial_molecule']), 3))
            for ii, grad in enumerate(parsed_gradients):
                for jj in range(int(len(grad) / 3)):
                    for kk in range(grad_format_length):
                        if grad[jj * 3][kk] != 'None':
                            sorted_gradients[ii][jj * grad_format_length + kk][0] = grad[jj * 3][kk]
                            sorted_gradients[ii][jj * grad_format_length + kk][1] = grad[jj * 3 + 1][kk]
                            sorted_gradients[ii][jj * grad_format_length + kk][2] = grad[jj * 3 + 2][kk]
            self.data['CDS_gradients'] = sorted_gradients
        else:
            self.data['CDS_gradients'] = None