import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def _set_post_define(self):
    """non-define keys, user-specified changes in the control file"""
    for p in list(self.parameters.keys()):
        if p in list(self.parameter_no_define.keys()):
            if self.parameter_no_define[p]:
                if self.parameters[p]:
                    if p in list(self.parameter_mapping.keys()):
                        fun = self.parameter_mapping[p]['to_control']
                        val = fun(self.parameters[p])
                    else:
                        val = self.parameters[p]
                    delete_data_group(self.parameter_group[p])
                    add_data_group(self.parameter_group[p], str(val))
                else:
                    delete_data_group(self.parameter_group[p])
    if self.control_kdg:
        for dg in self.control_kdg:
            delete_data_group(dg)
    if self.control_input:
        for inp in self.control_input:
            add_data_group(inp, raw=True)
    if self.pcpot:
        self.set_point_charges()