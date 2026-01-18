from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import (

        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        