import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
@ExperimentalAPI
def to_device(self, device, framework='torch'):
    """TODO: transfer batch to given device as framework tensor."""
    if framework == 'torch':
        assert torch is not None
        for k, v in self.items():
            self[k] = convert_to_torch_tensor(v, device)
    else:
        raise NotImplementedError
    return self