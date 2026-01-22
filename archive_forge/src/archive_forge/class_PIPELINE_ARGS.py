import os, sys
import numpy as np
import torch
from torch.nn import functional as F
class PIPELINE_ARGS:

    def __init__(self, temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2, alpha_decay=0.996, token_ban=[], token_stop=[], chunk_len=256):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency
        self.alpha_presence = alpha_presence
        self.alpha_decay = alpha_decay
        self.token_ban = token_ban
        self.token_stop = token_stop
        self.chunk_len = chunk_len