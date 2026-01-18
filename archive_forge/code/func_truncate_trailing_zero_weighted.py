import argparse
import os
import gc
import random
import ray
import orjson
import pyarrow
from pyarrow import parquet
def truncate_trailing_zero_weighted(tokens, weights):
    non_zero_index = len(weights) - 1
    while non_zero_index >= 0 and weights[non_zero_index] == 0:
        non_zero_index -= 1
    return (tokens[:non_zero_index + 1], weights[:non_zero_index + 1])