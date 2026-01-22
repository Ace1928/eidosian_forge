from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
apply activation checkpointing to model
    returns None as model is updated directly
    