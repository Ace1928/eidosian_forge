import argparse
import json
import os
import re
import sys
import types
import torch
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
def add_transformers_checkpoint_args(parser):
    parser.add_argument('--tokenizer_name', type=str, default=None, help='The name of the pre-trained tokenizer to save. If not None, the tokenizer will be saved. Only used when converting a Megatron checkpoint to a Transformers checkpoint.')
    parser.add_argument('--max_shard_size', type=str, default='10GB', help='The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). Only used when converting a Megatron checkpoint to a Transformers checkpoint.')
    return parser