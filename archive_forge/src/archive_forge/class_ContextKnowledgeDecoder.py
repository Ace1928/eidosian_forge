from typing import Dict
import numpy as np
import torch as th
import torch.nn as nn
from parlai.utils.torch import neginf
from parlai.agents.transformer.modules import TransformerGeneratorModel
class ContextKnowledgeDecoder(nn.Module):

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input, encoder_state, incr_state=None):
        encoder_output, encoder_mask, _ = encoder_state
        return self.transformer(input, (encoder_output, encoder_mask), incr_state)