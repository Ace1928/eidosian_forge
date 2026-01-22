import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
def _get_castep_version(castep_command, temp_dir):
    jname = 'dummy_jobname'
    stdout, stderr = ('', '')
    fallback_version = 16.0
    try:
        stdout, stderr = subprocess.Popen(castep_command.split() + ['--version'], stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=temp_dir, universal_newlines=True).communicate()
        if 'CASTEP version' not in stdout:
            stdout, stderr = subprocess.Popen(castep_command.split() + [jname], stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=temp_dir, universal_newlines=True).communicate()
    except Exception:
        msg = ''
        msg += 'Could not determine the version of your CASTEP binary \n'
        msg += 'This usually means one of the following \n'
        msg += '   * you do not have CASTEP installed \n'
        msg += '   * you have not set the CASTEP_COMMAND to call it \n'
        msg += '   * you have provided a wrong CASTEP_COMMAND. \n'
        msg += '     Make sure it is in your PATH\n\n'
        msg += stdout
        msg += stderr
        raise CastepVersionError(msg)
    if 'CASTEP version' in stdout:
        output_txt = stdout.split('\n')
        version_re = re.compile('CASTEP version:\\s*([0-9\\.]*)')
    else:
        output = open(os.path.join(temp_dir, '%s.castep' % jname))
        output_txt = output.readlines()
        output.close()
        version_re = re.compile('(?<=CASTEP version )[0-9.]*')
    for line in output_txt:
        if 'CASTEP version' in line:
            try:
                return float(version_re.findall(line)[0])
            except ValueError:
                return fallback_version