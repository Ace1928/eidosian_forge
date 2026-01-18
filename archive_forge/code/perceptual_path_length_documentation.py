import math
from typing import Literal, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torchmetrics.functional.image.lpips import _LPIPS
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE
Sample from the generator.

        Args:
            num_samples: Number of samples to generate.

        