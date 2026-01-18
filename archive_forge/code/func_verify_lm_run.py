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
def verify_lm_run(wps, golden_config, args):
    """Verify that words per second for a given benchmark run matches the golden data."""
    if torch.distributed.get_rank() == 0:
        logging.info('Throughput(wps) is {:.2f}.'.format(wps))
        if not wps > golden_config['avg_wps'] - 3 * golden_config['std_dev_wps']:
            raise RuntimeError('Throughput(wps):{:.2f} is below the golden threshold of an average value of {:.2f} and standard dev of {:.2f}.'.format(wps, golden_config['avg_wps'], golden_config['std_dev_wps']))
    for i in range(torch.cuda.device_count()):
        verify_peak_memory(i, golden_config, 1.1)