import io
import re
import shlex
import warnings
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any, Sequence
import collections.abc
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.io.cif_unicode import format_unicode, handle_subscripts
from ase.utils import iofunction
def _raw_scaled_positions(self) -> Optional[np.ndarray]:
    coords = [self.get(name) for name in ['_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']]
    if None in coords:
        return None
    return np.array(coords).T