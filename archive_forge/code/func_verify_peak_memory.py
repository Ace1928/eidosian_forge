import argparse
from collections import defaultdict
from functools import reduce
import gc
import logging
import math
import operator
import time
from datasets.wikitext2_data import get_real_dataloaders as get_real_wikitext2_dataloaders
from datasets.wikitext2_data import get_synthetic_dataloaders as get_synthetic_wikitext2_dataloaders
from models import transformer_lm
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from benchmarks.golden_configs.lm_wikitext2 import FSDP as lm_wikitext2
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
def verify_peak_memory(rank, golden_config, std_dev):
    logging.debug('Peak allocated bytes on cuda:0: {:1d}'.format(torch.cuda.memory_stats(rank)['allocated_bytes.all.peak']))
    current_device_usage = torch.cuda.memory_stats(rank)['allocated_bytes.all.peak']
    golden_ref = golden_config['peak_mem_usage'][rank]
    if not current_device_usage < golden_ref * std_dev:
        raise RuntimeError('Peak memory usage for cuda device {:d} is {:d} whichis less than golden reference value of {:d}'.format(rank, current_device_usage, golden_ref))