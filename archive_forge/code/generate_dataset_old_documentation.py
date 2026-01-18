from typing import Optional
from dataclasses import dataclass
import argparse
import json
import os
import random
import numpy as np
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from ray.util.multiprocessing import Pool

Generate training data based on conversations

Usage: python -m ochat.data.generate_data --in-file sharegpt_gpt4.json --tokenizer-name HF_REPO_NAME --out-dir .
