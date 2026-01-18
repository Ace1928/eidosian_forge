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
@pytest.mark.parametrize('gtype', [torch.float32, torch.float16], ids=['float', 'half'])
def test_percentile_clipping(gtype):
    gnorm_vec1 = torch.zeros(100, device='cuda')
    gnorm_vec2 = torch.zeros(100, device='cuda')
    n = 4
    step = 0
    percentile = 5
    for i in range(k):
        step += 1
        g = torch.randn(n, n, dtype=gtype, device='cuda')
        gnorm1, clip2, gnorm_scale = F.percentile_clipping(g, gnorm_vec2, step, percentile=percentile)
        assert gnorm_scale == 1.0 if gnorm1 < clip2 else clip2 / gnorm1
        gnorm2 = torch.norm(g.float())
        if step == 1:
            gnorm_vec1[:] = gnorm2
        else:
            gnorm_vec1[step % 100] = gnorm2
        vals, idx = torch.sort(gnorm_vec1)
        clip1 = vals[percentile]
        torch.testing.assert_close(gnorm_vec1, torch.sqrt(gnorm_vec2))
        torch.testing.assert_close(clip1, clip2)
        torch.testing.assert_close(gnorm1, gnorm2)