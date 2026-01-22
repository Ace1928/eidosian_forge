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
class VaspInput(dict, MSONable):
    """Class to contain a set of vasp input objects corresponding to a run."""

    def __init__(self, incar: dict | Incar, kpoints: Kpoints | None, poscar: Poscar, potcar: Potcar | None, optional_files: dict[PathLike, object] | None=None, **kwargs) -> None:
        """
        Initializes a VaspInput object with the given input files.

        Args:
            incar (Incar): The Incar object.
            kpoints (Kpoints): The Kpoints object.
            poscar (Poscar): The Poscar object.
            potcar (Potcar): The Potcar object.
            optional_files (dict): Other input files supplied as a dict of {filename: object}.
                The object should follow standard pymatgen conventions in implementing a
                as_dict() and from_dict method.
            **kwargs: Additional keyword arguments to be stored in the VaspInput object.
        """
        super().__init__(**kwargs)
        self.update({'INCAR': incar, 'KPOINTS': kpoints, 'POSCAR': poscar, 'POTCAR': potcar})
        if optional_files is not None:
            self.update(optional_files)

    def __str__(self):
        output = []
        for key, val in self.items():
            output.extend((key, str(val), ''))
        return '\n'.join(output)

    def as_dict(self):
        """MSONable dict."""
        dct = {key: val.as_dict() for key, val in self.items()}
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            VaspInput
        """
        sub_dct: dict[str, dict] = {'optional_files': {}}
        for key, val in dct.items():
            if key in ['INCAR', 'POSCAR', 'POTCAR', 'KPOINTS']:
                sub_dct[key.lower()] = MontyDecoder().process_decoded(val)
            elif key not in ['@module', '@class']:
                sub_dct['optional_files'][key] = MontyDecoder().process_decoded(val)
        return cls(**sub_dct)

    def write_input(self, output_dir='.', make_dir_if_not_present=True):
        """
        Write VASP input to a directory.

        Args:
            output_dir (str): Directory to write to. Defaults to current
                directory (".").
            make_dir_if_not_present (bool): Create the directory if not
                present. Defaults to True.
        """
        if make_dir_if_not_present:
            os.makedirs(output_dir, exist_ok=True)
        for k, v in self.items():
            if v is not None:
                with zopen(os.path.join(output_dir, k), mode='wt') as file:
                    file.write(str(v))

    @classmethod
    def from_directory(cls, input_dir: str, optional_files: dict | None=None) -> Self:
        """
        Read in a set of VASP input from a directory. Note that only the
        standard INCAR, POSCAR, POTCAR and KPOINTS files are read unless
        optional_filenames is specified.

        Args:
            input_dir (str): Directory to read VASP input from.
            optional_files (dict): Optional files to read in as well as a
                dict of {filename: Object type}. Object type must have a
                static method from_file.
        """
        sub_dct = {}
        for fname, ftype in [('INCAR', Incar), ('KPOINTS', Kpoints), ('POSCAR', Poscar), ('POTCAR', Potcar)]:
            try:
                full_zpath = zpath(os.path.join(input_dir, fname))
                sub_dct[fname.lower()] = ftype.from_file(full_zpath)
            except FileNotFoundError:
                sub_dct[fname.lower()] = None
        sub_dct['optional_files'] = {fname: ftype.from_file(os.path.join(input_dir, fname)) for fname, ftype in (optional_files or {}).items()}
        return cls(**sub_dct)

    def copy(self, deep: bool=True):
        """Deep copy of VaspInput."""
        if deep:
            return self.from_dict(self.as_dict())
        return type(self)(**{key.lower(): val for key, val in self.items()})

    def run_vasp(self, run_dir: PathLike='.', vasp_cmd: list | None=None, output_file: PathLike='vasp.out', err_file: PathLike='vasp.err') -> None:
        """
        Write input files and run VASP.

        Args:
            run_dir: Where to write input files and do the run.
            vasp_cmd: Args to be supplied to run VASP. Otherwise, the
                PMG_VASP_EXE in .pmgrc.yaml is used.
            output_file: File to write output.
            err_file: File to write err.
        """
        self.write_input(output_dir=run_dir)
        vasp_cmd = vasp_cmd or SETTINGS.get('PMG_VASP_EXE')
        if not vasp_cmd:
            raise ValueError('No VASP executable specified!')
        vasp_cmd = [os.path.expanduser(os.path.expandvars(t)) for t in vasp_cmd]
        if not vasp_cmd:
            raise RuntimeError('You need to supply vasp_cmd or set the PMG_VASP_EXE in .pmgrc.yaml to run VASP.')
        with cd(run_dir), open(output_file, mode='w', encoding='utf-8') as stdout_file, open(err_file, mode='w', encoding='utf-8', buffering=1) as stderr_file:
            subprocess.check_call(vasp_cmd, stdout=stdout_file, stderr=stderr_file)