import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def write_energy_files(self):
    """write input files for gromacs force and energy calculations
        for gromacs program energy"""
    filename = 'inputGenergy.txt'
    with open(filename, 'w') as output:
        output.write('Potential  \n')
        output.write('   \n')
        output.write('   \n')
    filename = 'inputGtraj.txt'
    with open(filename, 'w') as output:
        output.write('System  \n')
        output.write('   \n')
        output.write('   \n')