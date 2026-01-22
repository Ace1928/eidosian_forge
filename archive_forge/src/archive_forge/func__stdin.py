from subprocess import Popen, PIPE
from ase.calculators.calculator import Calculator
from ase.io import read
from .create_input import GenerateVaspInput
import time
import os
import sys
def _stdin(self, text, ending='\n'):
    if self.txt is not None:
        self.txt.write(text + ending)
    if self.print_log:
        print(text, end=ending)
    self.process.stdin.write(text + ending)
    if sys.version_info[0] >= 3:
        self.process.stdin.flush()