import torch
from typing import List, Optional, Dict
from vllm.worker.worker import Worker
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import SequenceGroupMetadata, SequenceData
from vllm.sampling_params import SamplingParams
from vllm.worker.cache_engine import CacheEngine
from vllm.model_executor.utils import set_random_seed
from dataclasses import dataclass, fields
@dataclass
class ExecuteModelData:
    """Helper data structure which facilitates cleaner tests.
    """
    seq_group_metadata_list: List[SequenceGroupMetadata]
    blocks_to_swap_in: Dict[int, int]
    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]

    def to_dict(self):
        return dict(((field.name, getattr(self, field.name)) for field in fields(self)))