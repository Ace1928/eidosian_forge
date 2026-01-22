import os
import os.path
from warnings import warn
from subprocess import Popen, PIPE
import numpy as np
import ase.io
from ase.units import Rydberg
from ase.calculators.calculator import (Calculator, all_changes, Parameters,
def _create_force_env(self):
    """Instantiates a new force-environment"""
    assert self._force_env_id is None
    label_dir = os.path.dirname(self.label)
    if len(label_dir) > 0 and (not os.path.exists(label_dir)):
        print('Creating directory: ' + label_dir)
        os.makedirs(label_dir)
    inp = self._generate_input()
    inp_fn = self.label + '.inp'
    out_fn = self.label + '.out'
    self._write_file(inp_fn, inp)
    self._shell.send('LOAD %s %s' % (inp_fn, out_fn))
    self._force_env_id = int(self._shell.recv())
    assert self._force_env_id > 0
    self._shell.expect('* READY')