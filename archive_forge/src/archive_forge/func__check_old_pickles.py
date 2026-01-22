from math import pi, sqrt, log
import sys
import numpy as np
from pathlib import Path
import ase.units as units
import ase.io
from ase.parallel import world, paropen
from ase.utils.filecache import get_json_cache
from .data import VibrationsData
from collections import namedtuple
def _check_old_pickles(self):
    from pathlib import Path
    eq_pickle_path = Path(f'{self.name}.eq.pckl')
    pickle2json_instructions = f'Found old pickle files such as {eq_pickle_path}.  Please remove them and recalculate or run "python -m ase.vibrations.pickle2json --help".'
    if len(self.cache) == 0 and eq_pickle_path.exists():
        raise RuntimeError(pickle2json_instructions)