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
class PercentSlice(metaclass=PercentSliceMeta):
    """Syntactic sugar for defining slice subsplits: `datasets.percent[75:-5]`.

    See the
    [guide on splits](../loading#slice-splits)
    for more information.
    """
    pass