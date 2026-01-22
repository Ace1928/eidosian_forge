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
class AtomicDisplacements:

    def _disp(self, a, i, step):
        if isinstance(i, str):
            i = 'xyz'.index(i)
        return Displacement(a, i, np.sign(step), abs(step), self)

    def _eq_disp(self):
        return self._disp(0, 0, 0)

    @property
    def ndof(self):
        return 3 * len(self.indices)