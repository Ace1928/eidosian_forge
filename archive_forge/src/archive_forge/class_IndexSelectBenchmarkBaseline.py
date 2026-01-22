import random
import torch
from utils import DTYPE2STR, benchmark_main_helper2, product_dict
import xformers.ops as xops
class IndexSelectBenchmarkBaseline(IndexSelectBenchmark):

    def fw(self) -> None:
        self.out = torch.cat([s[i].flatten() for s, i in zip(self.sources, self.indices)], dim=0)