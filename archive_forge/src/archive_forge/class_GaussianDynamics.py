import os
import copy
from collections.abc import Iterable
from shutil import which
from typing import Dict, Optional
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError
class GaussianDynamics:
    calctype = 'optimizer'
    delete = ['force']
    keyword: Optional[str] = None
    special_keywords: Dict[str, str] = dict()

    def __init__(self, atoms, calc=None):
        self.atoms = atoms
        if calc is not None:
            self.calc = calc
        else:
            if self.atoms.calc is None:
                raise ValueError('{} requires a valid Gaussian calculator object!'.format(self.__class__.__name__))
            self.calc = self.atoms.calc

    def todict(self):
        return {'type': self.calctype, 'optimizer': self.__class__.__name__}

    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)

    def set_keywords(self, kwargs):
        args = kwargs.pop(self.keyword, [])
        if isinstance(args, str):
            args = [args]
        elif isinstance(args, Iterable):
            args = list(args)
        for key, template in self.special_keywords.items():
            if key in kwargs:
                val = kwargs.pop(key)
                args.append(template.format(val))
        kwargs[self.keyword] = args

    def run(self, **kwargs):
        calc_old = self.atoms.calc
        params_old = copy.deepcopy(self.calc.parameters)
        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)
        self.set_keywords(kwargs)
        self.calc.set(**kwargs)
        self.atoms.calc = self.calc
        try:
            self.atoms.get_potential_energy()
        except OSError:
            converged = False
        else:
            converged = True
        atoms = read(self.calc.label + '.log')
        self.atoms.cell = atoms.cell
        self.atoms.positions = atoms.positions
        self.calc.parameters = params_old
        self.calc.reset()
        if calc_old is not None:
            self.atoms.calc = calc_old
        return converged