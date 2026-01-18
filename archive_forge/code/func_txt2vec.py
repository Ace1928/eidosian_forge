from parlai.core.torch_agent import TorchAgent, Output
import torch
from parlai.core.agents import Agent
def txt2vec(self, txt):
    """
        Return index of special tokens or range from 1 for each token.
        """
    self.idx = 0
    return [self[tok] for tok in txt.split()]