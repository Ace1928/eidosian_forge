import json
import math
import os
import re
from typing import Dict, List, Optional, Set
import torch
import torch.utils.benchmark as benchmark
from torch._C._profiler import (
from torch.profiler import profile
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
class FP32MatMulPattern(Pattern):

    def __init__(self, prof: profile, should_benchmark: bool=False):
        super().__init__(prof, should_benchmark)
        self.name = 'FP32 MatMul Pattern'
        self.description = "You are currently using GPU that supports TF32. Please enable TF32 by setting 'torch.backends.cuda.matmul.allow_tf32 = True'"
        self.url = 'https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices'

    @property
    def skip(self):
        if torch.version.hip is not None:
            has_tf32 = False
        else:
            has_tf32 = all((int(arch[3:]) >= 80 for arch in torch.cuda.get_arch_list()))
        return has_tf32 is False or super().skip or (not self.prof.record_shapes)

    def match(self, event: _ProfilerEvent):
        if event.tag != _EventType.TorchOp:
            return False
        assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
        if event.name == 'aten::mm':
            if event.extra_fields.allow_tf32_cublas is False:
                return True
        return False

    def report(self, event: _ProfilerEvent):
        return self.description

    def benchmark(self, events: List[_ProfilerEvent]):
        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        for shape in shapes_factor_map:
            matrixA = torch.randn(shape[0], device='cuda', dtype=torch.float32)
            matrixB = torch.randn(shape[1], device='cuda', dtype=torch.float32)
            fp32_timer = benchmark.Timer(stmt='torch.mm(matrixA, matrixB)', globals={'matrixA': matrixA, 'matrixB': matrixB})
            tf32_timer = benchmark.Timer(stmt='torch.mm(matrixA, matrixB)', setup='torch.backends.cuda.matmul.allow_tf32 = True', globals={'matrixA': matrixA, 'matrixB': matrixB})
            torch.backends.cuda.matmul.allow_tf32 = False
            fp32_time = fp32_timer.timeit(10).mean
            tf32_time = tf32_timer.timeit(10).mean
            shapes_factor_map[shape] = tf32_time / fp32_time
        return shapes_factor_map