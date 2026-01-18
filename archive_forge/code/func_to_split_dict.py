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
def to_split_dict(self):
    """Returns a list of SplitInfo protos that we have."""
    out = []
    for split_name, split_info in self.items():
        split_info = copy.deepcopy(split_info)
        split_info.name = split_name
        out.append(split_info)
    return out