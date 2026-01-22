from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
def _match_args(args1: Union[List, Tuple], args2: Union[List, Tuple]) -> bool:
    if len(args1) != len(args2):
        return False
    for a1, a2 in zip(args1, args2):
        if isinstance(a1, Node) and isinstance(a2, Node):
            matched = self._match_nodes(a1, a2, match)
        elif isinstance(a1, (list, tuple)) and isinstance(a2, (list, tuple)):
            matched = _match_args(a1, a2)
        else:
            matched = self._match_literals(a1, a2, match) or self.ignore_literals
        if not matched:
            return False
    return True