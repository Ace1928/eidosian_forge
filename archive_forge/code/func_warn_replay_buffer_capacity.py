import logging
import psutil
from typing import Optional, Any
import numpy as np
from ray.rllib.utils import deprecation_warning
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.replay_buffers import (
from ray.rllib.policy.sample_batch import concat_samples, MultiAgentBatch, SampleBatch
from ray.rllib.utils.typing import ResultDict, SampleBatchType, AlgorithmConfigDict
from ray.util import log_once
@DeveloperAPI
def warn_replay_buffer_capacity(*, item: SampleBatchType, capacity: int) -> None:
    """Warn if the configured replay buffer capacity is too large for machine's memory.

    Args:
        item: A (example) item that's supposed to be added to the buffer.
            This is used to compute the overall memory footprint estimate for the
            buffer.
        capacity: The capacity value of the buffer. This is interpreted as the
            number of items (such as given `item`) that will eventually be stored in
            the buffer.

    Raises:
        ValueError: If computed memory footprint for the buffer exceeds the machine's
            RAM.
    """
    if log_once('warn_replay_buffer_capacity'):
        item_size = item.size_bytes()
        psutil_mem = psutil.virtual_memory()
        total_gb = psutil_mem.total / 1000000000.0
        mem_size = capacity * item_size / 1000000000.0
        msg = 'Estimated max memory usage for replay buffer is {} GB ({} batches of size {}, {} bytes each), available system memory is {} GB'.format(mem_size, capacity, item.count, item_size, total_gb)
        if mem_size > total_gb:
            raise ValueError(msg)
        elif mem_size > 0.2 * total_gb:
            logger.warning(msg)
        else:
            logger.info(msg)