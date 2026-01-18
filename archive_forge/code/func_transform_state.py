import argparse
import json
import os.path
from collections import OrderedDict
import numpy as np
import requests
import torch
from flax.training.checkpoints import restore_checkpoint
from huggingface_hub import hf_hub_download
from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor
from transformers.image_utils import PILImageResampling
def transform_state(state_dict, classification_head=False):
    transformer_layers = get_n_layers(state_dict)
    new_state = OrderedDict()
    new_state['layernorm.bias'] = state_dict['optimizer']['target']['Transformer']['encoder_norm']['bias']
    new_state['layernorm.weight'] = state_dict['optimizer']['target']['Transformer']['encoder_norm']['scale']
    new_state['embeddings.patch_embeddings.projection.weight'] = np.transpose(state_dict['optimizer']['target']['embedding']['kernel'], (4, 3, 0, 1, 2))
    new_state['embeddings.patch_embeddings.projection.bias'] = state_dict['optimizer']['target']['embedding']['bias']
    new_state['embeddings.cls_token'] = state_dict['optimizer']['target']['cls']
    new_state['embeddings.position_embeddings'] = state_dict['optimizer']['target']['Transformer']['posembed_input']['pos_embedding']
    for i in range(transformer_layers):
        new_state.update(transform_state_encoder_block(state_dict, i))
    if classification_head:
        new_state = {'vivit.' + k: v for k, v in new_state.items()}
        new_state['classifier.weight'] = np.transpose(state_dict['optimizer']['target']['output_projection']['kernel'])
        new_state['classifier.bias'] = np.transpose(state_dict['optimizer']['target']['output_projection']['bias'])
    return {k: torch.tensor(v) for k, v in new_state.items()}