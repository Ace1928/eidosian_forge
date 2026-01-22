import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer

            Local helper function for _gather_flat_grad.
            Returns a flattened view of the input tensor.
            