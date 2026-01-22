import abc
import torch
from typing import Optional, Tuple, List, Any, Dict
from ...sparsifier import base_sparsifier
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier import utils
from torch.nn.utils import parametrize
import sys
import warnings
Detaches some data from the sparsifier.

        Args:
            name (str)
                Name of the data to be removed from the sparsifier

        Note:
            Currently private. Kind of used as a helper function when replacing data of the same name
        