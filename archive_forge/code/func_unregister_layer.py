from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
def unregister_layer(self, name):
    """Detaches the sparsifier from the layer
        """
    self.data_groups[name]['hook'].remove()
    self.state.pop(name)
    self.data_groups.pop(name)