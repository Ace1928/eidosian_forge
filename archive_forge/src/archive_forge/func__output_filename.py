import re
import ase.io.abinit as io
from ase.calculators.calculator import FileIOCalculator
from subprocess import check_output
def _output_filename(self):
    if self.v8_legacy_format:
        ext = '.txt'
    else:
        ext = '.abo'
    return self.label + ext