from __future__ import annotations
import itertools
import math
import os
import subprocess
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import PeriodicSite, Species, Structure
from pymatgen.util.coord import in_coord_list
def annotate_pick(obj, event):
    if picker.GetCellId() < 0:
        follower.VisibilityOff()
    else:
        pick_pos = picker.GetPickPosition()
        mapper = picker.GetMapper()
        if mapper in self.mapper_map:
            site = self.mapper_map[mapper]
            output = [site.species_string, 'Frac. coords: ' + ' '.join((f'{c:.4f}' for c in site.frac_coords))]
            source.SetText('\n'.join(output))
            follower.SetPosition(pick_pos)
            follower.VisibilityOn()