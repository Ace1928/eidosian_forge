from typing import Dict, List, Optional, Tuple
import torch
import torch.optim._functional as F
from torch import Tensor

        Similar to step, but operates on a single parameter and optionally a
        gradient tensor.
        