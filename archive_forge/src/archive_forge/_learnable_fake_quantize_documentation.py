import torch
from torch.nn.parameter import Parameter
from typing import List
Enable accumulation of data without updating quantization parameters.

        Enables static observer accumulating data from input but doesn't
        update the quantization parameters. Forward path returns the original X.
        