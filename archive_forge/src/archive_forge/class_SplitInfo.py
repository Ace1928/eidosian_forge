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
@dataclass
class SplitInfo:
    name: str = dataclasses.field(default='', metadata={'include_in_asdict_even_if_is_default': True})
    num_bytes: int = dataclasses.field(default=0, metadata={'include_in_asdict_even_if_is_default': True})
    num_examples: int = dataclasses.field(default=0, metadata={'include_in_asdict_even_if_is_default': True})
    shard_lengths: Optional[List[int]] = None
    dataset_name: Optional[str] = dataclasses.field(default=None, metadata={'include_in_asdict_even_if_is_default': True})

    @property
    def file_instructions(self):
        """Returns the list of dict(filename, take, skip)."""
        instructions = make_file_instructions(name=self.dataset_name, split_infos=[self], instruction=str(self.name))
        return instructions.file_instructions