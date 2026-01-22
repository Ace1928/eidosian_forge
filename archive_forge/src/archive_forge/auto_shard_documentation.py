import logging
from typing import Dict, List, Set
import torch
import torch.fx
from torch.fx.node import Node
Initializes a new _ExtendedLeafTracer object.

        Args:
            leaf_modules: The set of extra nn.Modules instances which will not be traced
                through but instead considered to be leaves.
        