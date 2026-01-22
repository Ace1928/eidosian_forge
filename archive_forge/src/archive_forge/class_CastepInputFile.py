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
class CastepInputFile:
    """Master class for CastepParam and CastepCell to inherit from"""
    _keyword_conflicts: List[Set[str]] = []

    def __init__(self, options_dict=None, keyword_tolerance=1):
        object.__init__(self)
        if options_dict is None:
            options_dict = CastepOptionDict({})
        self._options = options_dict._options
        self.__dict__.update(self._options)
        self._perm = np.clip(keyword_tolerance, 0, 2)
        self._conflict_dict = {kw: set(cset).difference({kw}) for cset in self._keyword_conflicts for kw in cset}

    def __repr__(self):
        expr = ''
        is_default = True
        for key, option in sorted(self._options.items()):
            if option.value is not None:
                is_default = False
                expr += '%20s : %s\n' % (key, option.value)
        if is_default:
            expr = 'Default\n'
        expr += 'Keyword tolerance: {0}'.format(self._perm)
        return expr

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            self.__dict__[attr] = value
            return
        if attr not in self._options.keys():
            if self._perm > 0:
                is_str = isinstance(value, str)
                is_block = False
                if hasattr(value, '__getitem__') and (not is_str) or (is_str and len(value.split('\n')) > 1):
                    is_block = True
            if self._perm == 0:
                similars = difflib.get_close_matches(attr, self._options.keys())
                if similars:
                    raise UserWarning('Option "%s" not known! You mean "%s"?' % (attr, similars[0]))
                else:
                    raise UserWarning('Option "%s" is not known!' % attr)
            elif self._perm == 1:
                warnings.warn('Option "%s" is not known and will be added as a %s' % (attr, 'block' if is_block else 'string'))
            attr = attr.lower()
            opt = CastepOption(keyword=attr, level='Unknown', option_type='block' if is_block else 'string')
            self._options[attr] = opt
            self.__dict__[attr] = opt
        else:
            attr = attr.lower()
            opt = self._options[attr]
        if not opt.type.lower() == 'block' and isinstance(value, str):
            value = value.replace(':', ' ')
        attrparse = '_parse_%s' % attr.lower()
        if not value is None:
            cset = self._conflict_dict.get(attr.lower(), {})
            for c in cset:
                if c in self._options and self._options[c].value:
                    warnings.warn('option "{attr}" conflicts with "{conflict}" in calculator. Setting "{conflict}" to None.'.format(attr=attr, conflict=c))
                    self._options[c].value = None
        if hasattr(self, attrparse):
            self._options[attr].value = self.__getattribute__(attrparse)(value)
        else:
            self._options[attr].value = value

    def __getattr__(self, name):
        if name[0] == '_' or self._perm == 0:
            raise AttributeError()
        if self._perm == 1:
            warnings.warn('Option %s is not known, returning None' % name)
        return CastepOption(keyword='none', level='Unknown', option_type='string', value=None)

    def get_attr_dict(self, raw=False, types=False):
        """Settings that go into .param file in a traditional dict"""
        attrdict = {k: o.raw_value if raw else o.value for k, o in self._options.items() if o.value is not None}
        if types:
            for key, val in attrdict.items():
                attrdict[key] = (val, self._options[key].type)
        return attrdict