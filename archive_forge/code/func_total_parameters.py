import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
def total_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of parameters in the model.

    :param model:
        the model whose parameters we wish to count.

    :return:
        total number of parameters in the model.
    """
    return sum((p.numel() for p in model.parameters()))