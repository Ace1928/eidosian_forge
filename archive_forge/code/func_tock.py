from itertools import product
import math
import random
import time
import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
def tock(self, name='default', evict=True, print_ms=True):
    if name in self.ends:
        self.ends[name].record()
        torch.cuda.synchronize()
        ms = self.starts[name].elapsed_time(self.ends[name])
        if name not in self.agg:
            self.agg[name] = 0.0
        self.agg[name] += ms
        if evict:
            self.starts.pop(name)
            self.ends.pop(name)
    if print_ms and name in self.agg:
        print(f'{name} took: {self.agg[name] / 1000.0:.5f}s')
    return self.agg[name]