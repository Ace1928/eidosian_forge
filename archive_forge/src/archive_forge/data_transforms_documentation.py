from typing import Dict
import numpy as np
import torch
from . import residue_constants as rc
from .tensor_utils import tensor_tree_map, tree_map
Construct denser atom positions (14 dimensions instead of 37).