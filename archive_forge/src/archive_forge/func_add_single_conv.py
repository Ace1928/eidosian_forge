import argparse
import os
import gc
import random
import ray
import orjson
import pyarrow
from pyarrow import parquet
def add_single_conv(output, tokens, weights):
    tokens, weights = truncate_trailing_zero_weighted(tokens, weights)
    if not tokens:
        return
    length = len(tokens)
    labels = [t if w != 0 else PAD_TOKEN_ID for t, w in zip(tokens, weights)]
    results = {'total_length': length, 'seqlens': [length], 'nz_input_ids': tokens, 'nz_position_ids': list(range(length)), 'nz_shifted_label_ids': labels[1:] + [PAD_TOKEN_ID], 'nz_shifted_loss_weights': weights[1:] + [0.0]}
    results['num_seqs'] = sum(results['nz_shifted_loss_weights'])
    for k, v in results.items():
        output[k].append(v)