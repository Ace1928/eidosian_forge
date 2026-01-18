import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (

        Minimizing the model from node with name `start` to node with name `end` base
        on self.settings. Find culprits that causes FxNetMinimizerRunFuncError or
        FxNetMinimizerResultMismatchError errors.

        Args:
            start: The name of the node where we want to start minimizing. If set
                to None, then we'll start with the first node of the model.
            end: The name of the node where we want to terminate minimizing. If
                set to None, we'll end with the last node of the model.

        Returns:
            nodes: A list of nodes that causes FxNetMinimizerRunFuncError or
                FxNetMinimizerResultMismatchError errors during minimizing.
        