from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
Returns a low-rank 2D tensor of square_side_length * matrix_approximation_rank.