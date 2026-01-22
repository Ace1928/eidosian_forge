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
class CastepOption:
    """"A CASTEP option. It handles basic conversions from string to its value
    type."""
    default_convert_types = {'boolean (logical)': 'bool', 'defined': 'bool', 'string': 'str', 'integer': 'int', 'real': 'float', 'integer vector': 'int_vector', 'real vector': 'float_vector', 'physical': 'float_physical', 'block': 'block'}

    def __init__(self, keyword, level, option_type, value=None, docstring='No information available'):
        self.keyword = keyword
        self.level = level
        self.type = option_type
        self._value = value
        self.__doc__ = docstring

    @property
    def value(self):
        if self._value is not None:
            if self.type.lower() in ('integer vector', 'real vector', 'physical'):
                return ' '.join(map(str, self._value))
            elif self.type.lower() in ('boolean (logical)', 'defined'):
                return str(self._value).upper()
            else:
                return str(self._value)

    @property
    def raw_value(self):
        return self._value

    @value.setter
    def value(self, val):
        if val is None:
            self.clear()
            return
        ctype = self.default_convert_types.get(self.type.lower(), 'str')
        typeparse = '_parse_%s' % ctype
        try:
            self._value = getattr(self, typeparse)(val)
        except ValueError:
            raise ConversionError(ctype, self.keyword, val)

    def clear(self):
        """Reset the value of the option to None again"""
        self._value = None

    @staticmethod
    def _parse_bool(value):
        try:
            value = _tf_table[str(value).strip().title()]
        except (KeyError, ValueError):
            raise ValueError()
        return value

    @staticmethod
    def _parse_str(value):
        value = str(value)
        return value

    @staticmethod
    def _parse_int(value):
        value = int(value)
        return value

    @staticmethod
    def _parse_float(value):
        value = float(value)
        return value

    @staticmethod
    def _parse_int_vector(value):
        if isinstance(value, str):
            if ',' in value:
                value = value.replace(',', ' ')
            value = list(map(int, value.split()))
        value = np.array(value)
        if value.shape != (3,) or value.dtype != int:
            raise ValueError()
        return list(value)

    @staticmethod
    def _parse_float_vector(value):
        if isinstance(value, str):
            if ',' in value:
                value = value.replace(',', ' ')
            value = list(map(float, value.split()))
        value = np.array(value) * 1.0
        if value.shape != (3,) or value.dtype != float:
            raise ValueError()
        return list(value)

    @staticmethod
    def _parse_float_physical(value):
        if isinstance(value, str):
            value = value.split()
        try:
            l = len(value)
        except TypeError:
            l = 1
            value = [value]
        if l == 1:
            try:
                value = (float(value[0]), '')
            except (TypeError, ValueError):
                raise ValueError()
        elif l == 2:
            try:
                value = (float(value[0]), value[1])
            except (TypeError, ValueError, IndexError):
                raise ValueError()
        else:
            raise ValueError()
        return value

    @staticmethod
    def _parse_block(value):
        if isinstance(value, str):
            return value
        elif hasattr(value, '__getitem__'):
            return '\n'.join(value)
        else:
            raise ValueError()

    def __repr__(self):
        if self._value:
            expr = 'Option: {keyword}({type}, {level}):\n{_value}\n'.format(**self.__dict__)
        else:
            expr = 'Option: {keyword}[unset]({type}, {level})'.format(**self.__dict__)
        return expr

    def __eq__(self, other):
        if not isinstance(other, CastepOption):
            return False
        else:
            return self.__dict__ == other.__dict__