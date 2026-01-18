from __future__ import annotations
import os
import shutil
import subprocess
from glob import glob
from typing import TYPE_CHECKING, Literal
from urllib.request import urlretrieve
from monty.json import jsanitize
from monty.serialization import dumpfn, loadfn
from ruamel import yaml
from pymatgen.core import OLD_SETTINGS_FILE, SETTINGS_FILE, Element
from pymatgen.io.cp2k.inputs import GaussianTypeOrbitalBasisSet, GthPotential
from pymatgen.io.cp2k.utils import chunk
Handle configure command.