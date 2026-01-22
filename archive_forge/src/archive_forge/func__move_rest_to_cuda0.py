import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
def _move_rest_to_cuda0(self, parameter: torch.Tensor):
    if parameter.device.type == 'cpu':
        return parameter.to('cuda:0')
    else:
        return parameter