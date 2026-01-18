import argparse
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import queue
import subprocess
import tempfile
import textwrap
import numpy as np
import torch
from torch.utils.benchmark.op_fuzzers import unary
from torch.utils.benchmark import Timer, Measurement
from typing import Dict, Tuple, List
def test_source(envs):
    """Ensure that subprocess"""
    for env in envs:
        result = run(f'source activate {env}')
        if result.returncode != 0:
            raise ValueError(f'Failed to source environment `{env}`')