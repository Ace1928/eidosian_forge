from __future__ import annotations
import logging
import sys
from collections import namedtuple
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse
import requests
from tqdm import tqdm
from pymatgen.core import DummySpecies, Structure
from pymatgen.util.due import Doi, due
from pymatgen.util.provenance import StructureNL
def _get_comp(sp_dict):
    return {_sanitize_symbol(symbol): conc for symbol, conc in zip(sp_dict['chemical_symbols'], sp_dict['concentration'])}