from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
def _collate_input_value(self, field):
    """
        Collects the join item field values into a list or set value for
        the given field, as follows:

        - If the field trait is a Set, then the values are collected into
        a set.

        - Otherwise, the values are collected into a list which preserves
        the iterables order. If the ``unique`` flag is set, then duplicate
        values are removed but the iterables order is preserved.
        """
    val = [self._slot_value(field, idx) for idx in range(self._next_slot_index)]
    basetrait = self._interface.inputs.trait(field)
    if isinstance(basetrait.trait_type, traits.Set):
        return set(val)
    if self._unique:
        return list(OrderedDict.fromkeys(val))
    return val