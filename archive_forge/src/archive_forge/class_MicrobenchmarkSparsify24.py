from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from utils import DTYPE2STR, benchmark_main_helper2, product_dict
import xformers.ops as xops
class MicrobenchmarkSparsify24(MicrobenchmarkBase):

    def fw(self) -> torch.Tensor:
        xops.sparsify24(self.input)
        return self.input