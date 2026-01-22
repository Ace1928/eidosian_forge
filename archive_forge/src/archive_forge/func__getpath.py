import os
import numpy as np
from ase.io.octopus.input import (
from ase.io.octopus.output import read_eigenvalues_file, read_static_info
from ase.calculators.calculator import (
def _getpath(self, path, check=False):
    path = os.path.join(self.directory, path)
    if check:
        if not os.path.exists(path):
            raise OctopusIOError('No such file or directory: %s' % path)
    return path