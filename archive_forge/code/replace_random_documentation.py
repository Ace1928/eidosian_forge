import collections
import logging
import torch
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from .. import config, inductor_prims
from ..pattern_matcher import (
from ..virtualized import V

    Horizontally fuse all the seed generation on each device

        a = inductor_seed(dev)
        b = inductor_seed(dev)

    Becomes:
        seeds = inductor_seeds(2, dev)
        a = inductor_lookup_seed(seeds, 0)
        b = inductor_lookup_seed(seeds, 1)

    We do this because seed creation is entirely launch overhead bound.
    