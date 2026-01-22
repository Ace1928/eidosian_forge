import logging
from typing import List, Tuple
import torch
import torch.nn as nn
def _check_for_inf_or_nan(self) -> None:
    """
        For each layer, check if any of the parameters with a gradient attribute
        contain an inf/nan. If any of the parameters' gradient contain an inf/nan,
        then that layer's found_inf_or_nan attribute is set to True and all
        remaining parameters for that layer are skipped.
        """
    for elt in self.layer_info:
        elt.found_inf_or_nan = False
        for _, param in elt.layer.named_parameters():
            if hasattr(param, 'grad') and param.grad is not None:
                if torch.isinf(param.grad).any().item() or torch.isnan(param.grad).any().item():
                    elt.found_inf_or_nan = True
                    break