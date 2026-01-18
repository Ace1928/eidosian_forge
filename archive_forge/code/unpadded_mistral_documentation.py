from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.mistral.configuration_mistral import MistralConfig

        UnpaddedMistralRMSNorm is equivalent to T5LayerNorm
        