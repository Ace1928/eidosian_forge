import os, sys
import numpy as np
import torch
from torch.nn import functional as F
class ABC_TOKENIZER:

    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 2
        self.eos_token_id = 3

    def encode(self, text):
        ids = [ord(c) for c in text]
        return ids

    def decode(self, ids):
        txt = ''.join((chr(idx) if idx > self.eos_token_id else '' for idx in ids if idx != self.eos_token_id))
        return txt