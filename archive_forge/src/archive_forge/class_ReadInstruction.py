import copy
import math
import os
import re
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.contrib.concurrent import thread_map
from .download.download_config import DownloadConfig
from .naming import _split_re, filenames_for_dataset_split
from .table import InMemoryTable, MemoryMappedTable, Table, concat_tables
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import cached_path
class ReadInstruction:
    """Reading instruction for a dataset.

    Examples::

      # The following lines are equivalent:
      ds = datasets.load_dataset('mnist', split='test[:33%]')
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction.from_spec('test[:33%]'))
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction('test', to=33, unit='%'))
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction(
          'test', from_=0, to=33, unit='%'))

      # The following lines are equivalent:
      ds = datasets.load_dataset('mnist', split='test[:33%]+train[1:-1]')
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction.from_spec(
          'test[:33%]+train[1:-1]'))
      ds = datasets.load_dataset('mnist', split=(
          datasets.ReadInstruction('test', to=33, unit='%') +
          datasets.ReadInstruction('train', from_=1, to=-1, unit='abs')))

      # The following lines are equivalent:
      ds = datasets.load_dataset('mnist', split='test[:33%](pct1_dropremainder)')
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction.from_spec(
          'test[:33%](pct1_dropremainder)'))
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction(
          'test', from_=0, to=33, unit='%', rounding="pct1_dropremainder"))

      # 10-fold validation:
      tests = datasets.load_dataset(
          'mnist',
          [datasets.ReadInstruction('train', from_=k, to=k+10, unit='%')
          for k in range(0, 100, 10)])
      trains = datasets.load_dataset(
          'mnist',
          [datasets.ReadInstruction('train', to=k, unit='%') + datasets.ReadInstruction('train', from_=k+10, unit='%')
          for k in range(0, 100, 10)])

    """

    def _init(self, relative_instructions):
        self._relative_instructions = relative_instructions

    @classmethod
    def _read_instruction_from_relative_instructions(cls, relative_instructions):
        """Returns ReadInstruction obj initialized with relative_instructions."""
        result = cls.__new__(cls)
        result._init(relative_instructions)
        return result

    def __init__(self, split_name, rounding=None, from_=None, to=None, unit=None):
        """Initialize ReadInstruction.

        Args:
            split_name (str): name of the split to read. Eg: 'train'.
            rounding (str, optional): The rounding behaviour to use when percent slicing is
                used. Ignored when slicing with absolute indices.
                Possible values:
                 - 'closest' (default): The specified percentages are rounded to the
                     closest value. Use this if you want specified percents to be as
                     much exact as possible.
                 - 'pct1_dropremainder': the specified percentages are treated as
                     multiple of 1%. Use this option if you want consistency. Eg:
                         len(5%) == 5 * len(1%).
                     Using this option, one might not be able to use the full set of
                     examples, if the number of those is not a multiple of 100.
            from_ (int):
            to (int): alternative way of specifying slicing boundaries. If any of
                {from_, to, unit} argument is used, slicing cannot be specified as
                string.
            unit (str): optional, one of:
                '%': to set the slicing unit as percents of the split size.
                'abs': to set the slicing unit as absolute numbers.
        """
        self._init([_RelativeInstruction(split_name, from_, to, unit, rounding)])

    @classmethod
    def from_spec(cls, spec):
        """Creates a `ReadInstruction` instance out of a string spec.

        Args:
            spec (`str`):
                Split(s) + optional slice(s) to read + optional rounding
                if percents are used as the slicing unit. A slice can be specified,
                using absolute numbers (`int`) or percentages (`int`).

        Examples:

            ```
            test: test split.
            test + validation: test split + validation split.
            test[10:]: test split, minus its first 10 records.
            test[:10%]: first 10% records of test split.
            test[:20%](pct1_dropremainder): first 10% records, rounded with the pct1_dropremainder rounding.
            test[:-5%]+train[40%:60%]: first 95% of test + middle 20% of train.
            ```

        Returns:
            ReadInstruction instance.
        """
        spec = str(spec)
        subs = _ADDITION_SEP_RE.split(spec)
        if not subs:
            raise ValueError(f'No instructions could be built out of {spec}')
        instruction = _str_to_read_instruction(subs[0])
        return sum((_str_to_read_instruction(sub) for sub in subs[1:]), instruction)

    def to_spec(self):
        rel_instr_specs = []
        for rel_instr in self._relative_instructions:
            rel_instr_spec = rel_instr.splitname
            if rel_instr.from_ is not None or rel_instr.to is not None:
                from_ = rel_instr.from_
                to = rel_instr.to
                unit = rel_instr.unit
                rounding = rel_instr.rounding
                unit = unit if unit == '%' else ''
                from_ = str(from_) + unit if from_ is not None else ''
                to = str(to) + unit if to is not None else ''
                slice_str = f'[{from_}:{to}]'
                rounding_str = f'({rounding})' if unit == '%' and rounding is not None and (rounding != 'closest') else ''
                rel_instr_spec += slice_str + rounding_str
            rel_instr_specs.append(rel_instr_spec)
        return '+'.join(rel_instr_specs)

    def __add__(self, other):
        """Returns a new ReadInstruction obj, result of appending other to self."""
        if not isinstance(other, ReadInstruction):
            msg = 'ReadInstruction can only be added to another ReadInstruction obj.'
            raise TypeError(msg)
        self_ris = self._relative_instructions
        other_ris = other._relative_instructions
        if self_ris[0].unit != 'abs' and other_ris[0].unit != 'abs' and (self._relative_instructions[0].rounding != other_ris[0].rounding):
            raise ValueError('It is forbidden to sum ReadInstruction instances with different rounding values.')
        return self._read_instruction_from_relative_instructions(self_ris + other_ris)

    def __str__(self):
        return self.to_spec()

    def __repr__(self):
        return f'ReadInstruction({self._relative_instructions})'

    def to_absolute(self, name2len):
        """Translate instruction into a list of absolute instructions.

        Those absolute instructions are then to be added together.

        Args:
            name2len (`dict`):
                Associating split names to number of examples.

        Returns:
            list of _AbsoluteInstruction instances (corresponds to the + in spec).
        """
        return [_rel_to_abs_instr(rel_instr, name2len) for rel_instr in self._relative_instructions]