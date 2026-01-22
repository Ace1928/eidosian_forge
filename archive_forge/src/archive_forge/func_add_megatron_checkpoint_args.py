import argparse
import json
import os
import re
import sys
import types
import torch
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
def add_megatron_checkpoint_args(parser):
    parser.add_argument('--target_tensor_model_parallel_size', type=int, default=1, help='The tensor model parallel size of the converted checkpoint. Only used when converting a Transformers checkpoint to a Megatron checkpoint.')
    parser.add_argument('--target_pipeline_model_parallel_size', type=int, default=1, help='The pipeline model parallel size of the converted checkpoint. Only used when converting a Transformers checkpoint to a Megatron checkpoint.')
    parser.add_argument('--target_data_parallel_size', type=int, default=1, help='The data parallel size of the converted checkpoint. Only used when converting a Transformers checkpoint to a Megatron checkpoint.')
    parser.add_argument('--target_params_dtype', type=str, default='fp32', help='The dtype of the converted checkpoint. Only used when converting a Transformers checkpoint to a Megatron checkpoint.')
    parser.add_argument('--make_vocab_size_divisible_by', type=int, default=128, help='Pad the vocab size to be divisible by this value. This is added for computational efficieny reasons. Only used when converting a Transformers checkpoint to a Megatron checkpoint.')
    parser.add_argument('--use_distributed_optimizer', action='store_true', help='If True, use the distributed optimizer. Only used when converting a Transformers checkpoint to a Megatron checkpoint.')
    return parser