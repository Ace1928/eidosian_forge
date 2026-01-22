import argparse
import json
import os
import re
import sys
import types
import torch
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
def add_checkpointing_args(parser):
    parser.add_argument('--megatron-path', type=str, default=None, help='Base directory of Megatron repository')
    parser.add_argument('--convert_checkpoint_from_megatron_to_transformers', action='store_true', help='If True, convert a Megatron checkpoint to a Transformers checkpoint. If False, convert a Transformers checkpoint to a Megatron checkpoint.')
    parser.add_argument('--load_path', type=str, required=True, help='Path to the checkpoint to convert.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to the converted checkpoint.')
    parser.add_argument('--print-checkpoint-structure', action='store_true')
    return parser