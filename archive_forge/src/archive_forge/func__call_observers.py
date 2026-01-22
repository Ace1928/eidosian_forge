import os
import sys
import shutil
import time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import jsonio
from ase.io.ulm import open as ulmopen
from ase.parallel import paropen, world, barrier
from ase.calculators.singlepoint import (SinglePointCalculator,
def _call_observers(self, obs):
    """Call pre/post write observers."""
    for function, interval, args, kwargs in obs:
        if (self.nframes + 1) % interval == 0:
            function(*args, **kwargs)