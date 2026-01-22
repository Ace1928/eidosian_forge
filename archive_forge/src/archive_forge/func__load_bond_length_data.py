from __future__ import annotations
import collections
import json
import os
import warnings
from typing import TYPE_CHECKING
from pymatgen.core import Element
def _load_bond_length_data():
    """Loads bond length data from json file."""
    with open(os.path.join(os.path.dirname(__file__), 'bond_lengths.json')) as file:
        data = collections.defaultdict(dict)
        for row in json.load(file):
            els = sorted(row['elements'])
            data[tuple(els)][row['bond_order']] = row['length']
        return data