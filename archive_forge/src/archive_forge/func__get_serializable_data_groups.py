from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
def _get_serializable_data_groups(self):
    """Exclude hook and layer from the config keys before serializing

        TODO: Might have to treat functions (reduce_fn, mask_fn etc) in a different manner while serializing.
              For time-being, functions are treated the same way as other attributes
        """
    data_groups: Dict[str, Any] = defaultdict()
    for name, config in self.data_groups.items():
        new_config = {key: value for key, value in config.items() if key not in ['hook', 'layer']}
        data_groups[name] = new_config
    return data_groups