import argparse
import os
from pathlib import Path
import torch
from accelerate.utils.modeling import find_tied_parameters
from seamless_communication.models.inference.translator import Translator
from transformers import (
from transformers.utils import logging
def _load_hf_config(model_type='medium'):
    if model_type == 'medium':
        kwargs = {'vocab_size': 256206, 't2u_vocab_size': 10082, 'hidden_size': 1024, 'max_position_embeddings': 4096, 'encoder_layers': 12, 'decoder_layers': 12, 'encoder_ffn_dim': 4096, 'decoder_ffn_dim': 4096, 't2u_encoder_layers': 4, 't2u_decoder_layers': 4, 'speech_encoder_layers': 12}
        return SeamlessM4TConfig(**kwargs)
    else:
        return SeamlessM4TConfig()