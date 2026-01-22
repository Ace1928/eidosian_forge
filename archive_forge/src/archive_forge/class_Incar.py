from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
class Incar(dict, MSONable):
    """
    INCAR object for reading and writing INCAR files. Essentially consists of
    a dictionary with some helper functions.
    """

    def __init__(self, params: dict[str, Any] | None=None):
        """
        Creates an Incar object.

        Args:
            params (dict): A set of input parameters as a dictionary.
        """
        super().__init__()
        if params:
            if (params.get('MAGMOM') and isinstance(params['MAGMOM'][0], (int, float))) and (params.get('LSORBIT') or params.get('LNONCOLLINEAR')):
                val = []
                for idx in range(len(params['MAGMOM']) // 3):
                    val.append(params['MAGMOM'][idx * 3:(idx + 1) * 3])
                params['MAGMOM'] = val
            self.update(params)

    def __setitem__(self, key: str, val: Any):
        """
        Add parameter-val pair to Incar. Warns if parameter is not in list of
        valid INCAR tags. Also cleans the parameter and val by stripping
        leading and trailing white spaces.
        """
        super().__setitem__(key.strip(), Incar.proc_val(key.strip(), val.strip()) if isinstance(val, str) else val)

    def as_dict(self) -> dict:
        """MSONable dict."""
        dct = dict(self)
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        return dct

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """
        Args:
            dct (dict): Serialized Incar

        Returns:
            Incar
        """
        if dct.get('MAGMOM') and isinstance(dct['MAGMOM'][0], dict):
            dct['MAGMOM'] = [Magmom.from_dict(m) for m in dct['MAGMOM']]
        return Incar({k: v for k, v in dct.items() if k not in ('@module', '@class')})

    def copy(self):
        return type(self)(self)

    def get_str(self, sort_keys: bool=False, pretty: bool=False) -> str:
        """
        Returns a string representation of the INCAR. The reason why this
        method is different from the __str__ method is to provide options for
        pretty printing.

        Args:
            sort_keys (bool): Set to True to sort the INCAR parameters
                alphabetically. Defaults to False.
            pretty (bool): Set to True for pretty aligned output. Defaults
                to False.
        """
        keys = sorted(self) if sort_keys else list(self)
        lines = []
        for key in keys:
            if key == 'MAGMOM' and isinstance(self[key], list):
                value = []
                if isinstance(self[key][0], (list, Magmom)) and (self.get('LSORBIT') or self.get('LNONCOLLINEAR')):
                    value.append(' '.join((str(i) for j in self[key] for i in j)))
                elif self.get('LSORBIT') or self.get('LNONCOLLINEAR'):
                    for m, g in itertools.groupby(self[key]):
                        value.append(f'3*{len(tuple(g))}*{m}')
                else:
                    for m, g in itertools.groupby(self[key], key=float):
                        value.append(f'{len(tuple(g))}*{m}')
                lines.append([key, ' '.join(value)])
            elif isinstance(self[key], list):
                lines.append([key, ' '.join(map(str, self[key]))])
            else:
                lines.append([key, self[key]])
        if pretty:
            return str(tabulate([[line[0], '=', line[1]] for line in lines], tablefmt='plain'))
        return str_delimited(lines, None, ' = ') + '\n'

    def __str__(self):
        return self.get_str(sort_keys=True, pretty=False)

    def write_file(self, filename: PathLike):
        """Write Incar to a file.

        Args:
            filename (str): filename to write to.
        """
        with zopen(filename, mode='wt') as file:
            file.write(str(self))

    @classmethod
    def from_file(cls, filename: PathLike) -> Self:
        """Reads an Incar object from a file.

        Args:
            filename (str): Filename for file

        Returns:
            Incar object
        """
        with zopen(filename, mode='rt') as file:
            return cls.from_str(file.read())

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Reads an Incar object from a string.

        Args:
            string (str): Incar string

        Returns:
            Incar object
        """
        lines = list(clean_lines(string.splitlines()))
        params = {}
        for line in lines:
            for sline in line.split(';'):
                if (m := re.match('(\\w+)\\s*=\\s*(.*)', sline.strip())):
                    key = m.group(1).strip()
                    val = m.group(2).strip()
                    val = Incar.proc_val(key, val)
                    params[key] = val
        return cls(params)

    @staticmethod
    def proc_val(key: str, val: Any):
        """Helper method to convert INCAR parameters to proper types like ints, floats, lists, etc.

        Args:
            key: INCAR parameter key
            val: Actual value of INCAR parameter.
        """
        list_keys = ('LDAUU', 'LDAUL', 'LDAUJ', 'MAGMOM', 'DIPOL', 'LANGEVIN_GAMMA', 'QUAD_EFG', 'EINT')
        bool_keys = ('LDAU', 'LWAVE', 'LSCALU', 'LCHARG', 'LPLANE', 'LUSE_VDW', 'LHFCALC', 'ADDGRID', 'LSORBIT', 'LNONCOLLINEAR')
        float_keys = ('EDIFF', 'SIGMA', 'TIME', 'ENCUTFOCK', 'HFSCREEN', 'POTIM', 'EDIFFG', 'AGGAC', 'PARAM1', 'PARAM2')
        int_keys = ('NSW', 'NBANDS', 'NELMIN', 'ISIF', 'IBRION', 'ISPIN', 'ISTART', 'ICHARG', 'NELM', 'ISMEAR', 'NPAR', 'LDAUPRINT', 'LMAXMIX', 'ENCUT', 'NSIM', 'NKRED', 'NUPDOWN', 'ISPIND', 'LDAUTYPE', 'IVDW')
        lower_str_keys = ('ML_MODE',)

        def smart_int_or_float(num_str):
            if num_str.find('.') != -1 or num_str.lower().find('e') != -1:
                return float(num_str)
            return int(num_str)
        with contextlib.suppress(ValueError):
            if key in list_keys:
                output = []
                tokens = re.findall('(-?\\d+\\.?\\d*)\\*?(-?\\d+\\.?\\d*)?\\*?(-?\\d+\\.?\\d*)?', val)
                for tok in tokens:
                    if tok[2] and '3' in tok[0]:
                        output.extend([smart_int_or_float(tok[2])] * int(tok[0]) * int(tok[1]))
                    elif tok[1]:
                        output.extend([smart_int_or_float(tok[1])] * int(tok[0]))
                    else:
                        output.append(smart_int_or_float(tok[0]))
                return output
            if key in bool_keys:
                if (m := re.match('^\\.?([T|F|t|f])[A-Za-z]*\\.?', val)):
                    return m.group(1).lower() == 't'
                raise ValueError(f'{key} should be a boolean type!')
            if key in float_keys:
                return float(re.search('^-?\\d*\\.?\\d*[e|E]?-?\\d*', val).group(0))
            if key in int_keys:
                return int(re.match('^-?[0-9]+', val).group(0))
            if key in lower_str_keys:
                return val.strip().lower()
        with contextlib.suppress(ValueError):
            return int(val)
        with contextlib.suppress(ValueError):
            return float(val)
        if 'true' in val.lower():
            return True
        if 'false' in val.lower():
            return False
        return val.strip().capitalize()

    def diff(self, other: Incar) -> dict[str, dict[str, Any]]:
        """
        Diff function for Incar. Compares two Incars and indicates which
        parameters are the same and which are not. Useful for checking whether
        two runs were done using the same parameters.

        Args:
            other (Incar): The other Incar object to compare to.

        Returns:
            dict[str, dict]: of the following format:
                {"Same" : parameters_that_are_the_same, "Different": parameters_that_are_different}
                Note that the parameters are return as full dictionaries of values. E.g. {"ISIF":3}
        """
        similar_param = {}
        different_param = {}
        for k1, v1 in self.items():
            if k1 not in other:
                different_param[k1] = {'INCAR1': v1, 'INCAR2': None}
            elif v1 != other[k1]:
                different_param[k1] = {'INCAR1': v1, 'INCAR2': other[k1]}
            else:
                similar_param[k1] = v1
        for k2, v2 in other.items():
            if k2 not in similar_param and k2 not in different_param and (k2 not in self):
                different_param[k2] = {'INCAR1': None, 'INCAR2': v2}
        return {'Same': similar_param, 'Different': different_param}

    def __add__(self, other):
        """
        Add all the values of another INCAR object to this object.
        Facilitates the use of "standard" INCARs.
        """
        params = dict(self.items())
        for key, val in other.items():
            if key in self and val != self[key]:
                raise ValueError(f'Incars have conflicting values for {key}: {self[key]} != {val}')
            params[key] = val
        return Incar(params)

    def check_params(self) -> None:
        """Check INCAR for invalid tags or values.
        If a tag doesn't exist, calculation will still run, however VASP
        will ignore the tag and set it as default without letting you know.
        """
        with open(os.path.join(module_dir, 'incar_parameters.json'), encoding='utf-8') as json_file:
            incar_params = json.loads(json_file.read())
        for tag, val in self.items():
            if tag not in incar_params:
                warnings.warn(f'Cannot find {tag} in the list of INCAR tags', BadIncarWarning, stacklevel=2)
                continue
            param_type = incar_params[tag].get('type')
            allowed_values = incar_params[tag].get('values')
            if param_type is not None and type(val).__name__ != param_type:
                warnings.warn(f'{tag}: {val} is not a {param_type}', BadIncarWarning, stacklevel=2)
            if allowed_values is not None and val not in allowed_values:
                warnings.warn(f'{tag}: Cannot find {val} in the list of values', BadIncarWarning, stacklevel=2)