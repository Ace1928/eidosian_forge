from functools import lru_cache
import torch
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .modules import MemNN, opt_to_kwargs
@lru_cache(maxsize=None)
def _time_feature(self, i):
    """
        Return time feature token at specified index.
        """
    return '__tf{}__'.format(i)