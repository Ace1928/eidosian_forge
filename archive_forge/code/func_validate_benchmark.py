import argparse
from enum import Enum
import importlib
import logging
import tempfile
import time
from typing import Any, List, Optional, cast
from golden_configs import oss_mnist
import numpy as np
import torch
import torch.autograd.profiler as profiler
from torch.cuda.amp import GradScaler as TorchGradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from benchmarks.datasets.mnist import setup_cached_mnist
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
def validate_benchmark(measurements, final_loss, args, check_regression):
    """Validate the measurments against the golden benchmark config."""
    golden_data = oss_mnist.get_golden_real_stats()
    max_memory = -1.0
    rank = dist.get_rank()
    if not args.cpu:
        torch.cuda.synchronize(rank)
        max_memory = torch.cuda.max_memory_allocated(rank) / 2 ** 20
        logging.info(f'[{rank}] : Peak memory {max_memory:.1f}MiB')
    measurements.sort()
    median = measurements[len(measurements) // 2]
    abs_diff = list(map(lambda x: abs(x - median), measurements))
    abs_diff.sort()
    mad = abs_diff[len(measurements) // 2] if args.epochs > 2 else -1
    logging.info(f'[{rank}] : Median speed: {median:.2f} +/- {mad:.2f}')
    if check_regression and rank == 0:
        assert median + 8.0 * mad > golden_data['reference_speed'], f'Speed regression detected: {median + 8.0 * mad} vs.  {golden_data['reference_speed']}'
        assert max_memory < 1.05 * golden_data['reference_memory'], f'Memory use regression detected: {max_memory} vs. {1.05 * golden_data['reference_memory']}'
        assert cast(float, final_loss) - golden_data['reference_loss'] < 0.01, f'Loss regression detected: {final_loss} vs. {golden_data['reference_loss']}'
        logging.info('[Regression Test] VALID')