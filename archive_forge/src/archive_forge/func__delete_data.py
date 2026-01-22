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
def _delete_data(self, name):
    """Detaches some data from the sparsifier.

        Args:
            name (str)
                Name of the data to be removed from the sparsifier

        Note:
            Currently private. Kind of used as a helper function when replacing data of the same name
        """
    self.squash_mask(names=[name], leave_parametrized=False)
    delattr(self._container, name)
    self.state.pop(name)
    self.data_groups.pop(name)