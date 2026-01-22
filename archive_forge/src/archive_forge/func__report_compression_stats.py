from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def _report_compression_stats(bucket, state):
    """
    Report compression stats at the frequency of `compression_stats_logging_frequency` specified in PowerSGD state.
    """
    if bucket.is_last() and state.iter >= state.next_stats_report:
        stats = state.compression_stats()
        logger.info('Compression stats: iter %s, total before compression %s, total after compression %s, rate %s', state.iter, stats[1], stats[2], stats[0])
        state.next_stats_report = state.iter + state.compression_stats_logging_frequency