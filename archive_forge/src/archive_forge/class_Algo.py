from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
class Algo(Enum):
    FFT = 0
    DCT = 1