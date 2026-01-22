import functools
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple
import torch
from torch.testing._internal.composite_compliance import (
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
def _analyze_operators(function, *args) -> List[ProfileMetadata]:
    """
    Use ProfileOperatorsTorchDispatchMode to get runtime and memory info.

    Args:
        function: The function to optimize which will be selectively checkpointed. Usually the forward pass
            of the model.
        *args: Arguments to pass in to the given ``function``.

    Returns:
        A list of tuples, where each tuples contains the name of the operator, the runtime of the operator,
            and the memory usage of the operator.

    """
    profile_ops = ProfileOperatorsTorchDispatchMode()
    with profile_ops:
        function(*args)
    data = profile_ops.data
    return data