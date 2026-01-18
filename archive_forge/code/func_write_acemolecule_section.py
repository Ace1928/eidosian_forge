import os
from copy import deepcopy
from ase.io import read
from ase.calculators.calculator import ReadError
from ase.calculators.calculator import FileIOCalculator
def write_acemolecule_section(self, fpt, section, depth=0):
    """Write parameters in each section of input

        Parameters
        ==========
        fpt: ACE-Moleucle input file object. Should be write mode.
        section: Dictionary of a parameter section.
        depth: Nested input depth.
        """
    for section, section_param in section.items():
        if isinstance(section_param, str) or isinstance(section_param, int) or isinstance(section_param, float):
            fpt.write('    ' * depth + str(section) + ' ' + str(section_param) + '\n')
        else:
            if isinstance(section_param, dict):
                fpt.write('    ' * depth + '%% ' + str(section) + '\n')
                self.write_acemolecule_section(fpt, section_param, depth + 1)
                fpt.write('    ' * depth + '%% End\n')
            if isinstance(section_param, list):
                for val in section_param:
                    fpt.write('    ' * depth + str(section) + ' ' + str(val) + '\n')