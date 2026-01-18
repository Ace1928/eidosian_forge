import argparse
import os
import gc
import random
import ray
import orjson
import pyarrow
from pyarrow import parquet

Generate training data based on conversations

Usage: python -m ochat.data.generate_data --in-file sharegpt_gpt4.jsonl --tokenizer-name HF_REPO_NAME --out-dir .
