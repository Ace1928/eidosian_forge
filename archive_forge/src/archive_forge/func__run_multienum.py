from __future__ import annotations
import fractions
import itertools
import logging
import math
import re
import subprocess
from glob import glob
from shutil import which
from threading import Timer
import numpy as np
from monty.dev import requires
from monty.fractions import lcm
from monty.tempfile import ScratchDir
from pymatgen.core import DummySpecies, PeriodicSite, Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def _run_multienum(self):
    with subprocess.Popen([enum_cmd], stdout=subprocess.PIPE, stdin=subprocess.PIPE, close_fds=True) as p:
        if self.timeout:
            timed_out = False
            timer = Timer(self.timeout * 60, lambda p: p.kill(), [p])
            try:
                timer.start()
                output = p.communicate()[0].decode('utf-8')
            finally:
                if not timer.is_alive():
                    timed_out = True
                timer.cancel()
            if timed_out:
                raise TimeoutError('Enumeration took too long')
        else:
            output = p.communicate()[0].decode('utf-8')
    count = 0
    start_count = False
    for line in output.strip().split('\n'):
        if line.strip().endswith('RunTot'):
            start_count = True
        elif start_count and re.match('\\d+\\s+.*', line.strip()):
            count = int(line.split()[-1])
    logger.debug(f'Enumeration resulted in {count} structures')
    return count