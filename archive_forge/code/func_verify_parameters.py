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
def verify_parameters(self):
    """detect wrong or not implemented parameters"""
    if self.define_str is not None:
        assert isinstance(self.define_str, str)
        assert len(self.define_str) != 0
        return
    for par in self.parameters:
        assert par in self.parameter_spec, 'invalid parameter: ' + par
    if self.parameters['use dft']:
        func_list = [x.lower() for x in self.available_functionals]
        func = self.parameters['density functional']
        assert func.lower() in func_list, 'density functional not available / not supported'
    assert self.parameters['multiplicity'], 'multiplicity not defined'
    if self.parameters['rohf']:
        raise NotImplementedError('ROHF not implemented')
    if self.parameters['initial guess'] not in ['eht', 'hcore']:
        if not (isinstance(self.parameters['initial guess'], dict) and 'use' in self.parameters['initial guess'].keys()):
            raise ValueError('Wrong input for initial guess')
    if not self.parameters['use basis set library']:
        raise NotImplementedError('Explicit basis set definition')
    if self.parameters['point group'] != 'c1':
        raise NotImplementedError('Point group not impemeneted')