from pathlib import Path
from re import compile
import numpy as np
from ase import Atoms
from ase.utils import reader
from ase.units import Bohr
def _labelize(raw_label):
    return _label_strip_re.sub('', raw_label).lower()