from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from utils import DTYPE2STR, benchmark_main_helper2, product_dict
import xformers.ops as xops
class MicrobenchmarkBase:

    def __init__(self, B_in_hidden_out_ft: Tuple[int, int, int, int], dtype, bias: bool, bw: bool) -> None:
        B, in_ft, hid_ft, out_ft = B_in_hidden_out_ft
        super().__init__()
        self.label = 'mlp'
        self.sub_label = f'{DTYPE2STR[dtype]} ({B},{in_ft},{hid_ft},{out_ft}){(' b' if bias else '')}'
        self.input = torch.randn([B, in_ft], device='cuda', dtype=dtype, requires_grad=True)
        self.input_colMajor = self.input.t().contiguous().t()
        self.input_sp = xops.sparsify24(self.input)

    def bw(self) -> None:
        return None