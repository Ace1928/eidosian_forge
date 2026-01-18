from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
Simply returns max_num_seqs as num_gpu_blocks, 0 as num_cpu_blocks.