import random
import torch
from utils import DTYPE2STR, benchmark_main_helper2, product_dict
import xformers.ops as xops
class ScaledIndexAddBenchmark:

    def __init__(self, dtype, scaling: bool, shape, bw: bool) -> None:
        B_src, B_out, M, D = shape
        torch.manual_seed(B_out + B_src)
        dtype_str = DTYPE2STR.get(dtype, dtype)
        self.sub_label = f'{dtype_str} B_src={B_src}, B_out={B_out}, M={M}, D={D} s={('Y' if scaling else 'N')}'
        self.label = 'scaled_index_add'
        self.alpha = 0.73
        self.inp = torch.randn([B_out, M, D], device='cuda', dtype=dtype, requires_grad=bw)
        self.src = torch.randn([B_src, M, D], device='cuda', dtype=dtype, requires_grad=bw)
        self.scaling = torch.randn([D], device='cuda', dtype=dtype, requires_grad=bw) if scaling else None
        self.index = torch.tensor([i for i in range(self.src.shape[0])], dtype=torch.int64, device='cuda')
        self.grad = torch.randn([B_out, M, D], device='cuda', dtype=dtype)
        self.out = torch.Tensor()

    def fw(self) -> None:
        self.out = xops.scaled_index_add(input=self.inp.clone(), index=self.index, source=self.src, scaling=self.scaling, alpha=self.alpha)

    def bw(self):
        self.inp.grad = None
        self.src.grad = None
        if self.scaling is not None:
            self.scaling.grad = None
        self.out.backward(self.grad, retain_graph=True)