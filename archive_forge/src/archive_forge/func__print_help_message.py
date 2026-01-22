from __future__ import annotations
import itertools
import json
import logging
import math
import os
import platform
import re
import sys
import warnings
from enum import Enum, unique
from time import sleep
from typing import TYPE_CHECKING, Any, Literal
import requests
from monty.json import MontyDecoder, MontyEncoder
from ruamel.yaml import YAML
from tqdm import tqdm
from pymatgen.core import SETTINGS, Composition, Element, Structure
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.exp_entries import ExpEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
@staticmethod
def _print_help_message(nomad_exist_task_ids, task_ids, file_patterns, task_types):
    non_exist_ids = set(task_ids) - set(nomad_exist_task_ids)
    warnings.warn(f'For file_patterns={file_patterns!r}] and task_types={task_types!r}, \nthe following ids are not found on NOMAD [{list(non_exist_ids)}]. \nIf you need to upload them, please contact Patrick Huck at phuck@lbl.gov')