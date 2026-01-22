import abc
import collections
import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from .arrow_reader import FileInstructions, make_file_instructions
from .naming import _split_re
from .utils.py_utils import NonMutableDict, asdict
class NamedSplit(SplitBase):
    """Descriptor corresponding to a named split (train, test, ...).

    Example:
        Each descriptor can be composed with other using addition or slice:

            ```py
            split = datasets.Split.TRAIN.subsplit(datasets.percent[0:25]) + datasets.Split.TEST
            ```

        The resulting split will correspond to 25% of the train split merged with
        100% of the test split.

        A split cannot be added twice, so the following will fail:

            ```py
            split = (
                    datasets.Split.TRAIN.subsplit(datasets.percent[:25]) +
                    datasets.Split.TRAIN.subsplit(datasets.percent[75:])
            )  # Error
            split = datasets.Split.TEST + datasets.Split.ALL  # Error
            ```

        The slices can be applied only one time. So the following are valid:

            ```py
            split = (
                    datasets.Split.TRAIN.subsplit(datasets.percent[:25]) +
                    datasets.Split.TEST.subsplit(datasets.percent[:50])
            )
            split = (datasets.Split.TRAIN + datasets.Split.TEST).subsplit(datasets.percent[:50])
            ```

        But this is not valid:

            ```py
            train = datasets.Split.TRAIN
            test = datasets.Split.TEST
            split = train.subsplit(datasets.percent[:25]).subsplit(datasets.percent[:25])
            split = (train.subsplit(datasets.percent[:25]) + test).subsplit(datasets.percent[:50])
            ```
    """

    def __init__(self, name):
        self._name = name
        split_names_from_instruction = [split_instruction.split('[')[0] for split_instruction in name.split('+')]
        for split_name in split_names_from_instruction:
            if not re.match(_split_re, split_name):
                raise ValueError(f"Split name should match '{_split_re}' but got '{split_name}'.")

    def __str__(self):
        return self._name

    def __repr__(self):
        return f'NamedSplit({self._name!r})'

    def __eq__(self, other):
        """Equality: datasets.Split.TRAIN == 'train'."""
        if isinstance(other, NamedSplit):
            return self._name == other._name
        elif isinstance(other, SplitBase):
            return False
        elif isinstance(other, str):
            return self._name == other
        else:
            raise ValueError(f'Equality not supported between split {self} and {other}')

    def __lt__(self, other):
        return self._name < other._name

    def __hash__(self):
        return hash(self._name)

    def get_read_instruction(self, split_dict):
        return SplitReadInstruction(split_dict[self._name])