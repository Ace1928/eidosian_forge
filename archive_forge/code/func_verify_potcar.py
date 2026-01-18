from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
def verify_potcar(self) -> tuple[bool, bool]:
    """
        Attempts to verify the integrity of the POTCAR data.

        This method checks the whole file (removing only the SHA256
        metadata) against the SHA256 hash in the header if this is found.
        If no SHA256 hash is found in the file, the file hash (md5 hash of the
        whole file) is checked against all POTCAR file hashes known to pymatgen.

        Returns:
            tuple[bool, bool]: has_sha256 and passed_hash_check are returned.
        """
    if self.hash_sha256_from_file:
        has_sha256 = True
        hash_is_valid = self.hash_sha256_from_file == self.sha256_computed_file_hash
    else:
        has_sha256 = False
        md5_file_hash = self.md5_computed_file_hash
        hash_is_valid = md5_file_hash in VASP_POTCAR_HASHES
    return (has_sha256, hash_is_valid)