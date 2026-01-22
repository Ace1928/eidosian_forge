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
class SubSplitInfo:
    """Wrapper around a sub split info.
    This class expose info on the subsplit:
    ```
    ds, info = datasets.load_dataset(..., split='train[75%:]', with_info=True)
    info.splits['train[75%:]'].num_examples
    ```
    """
    instructions: FileInstructions

    @property
    def num_examples(self):
        """Returns the number of example in the subsplit."""
        return self.instructions.num_examples

    @property
    def file_instructions(self):
        """Returns the list of dict(filename, take, skip)."""
        return self.instructions.file_instructions