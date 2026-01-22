import copy
from collections import defaultdict
import os
import time
import pickle
import importlib
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
from vllm.utils import (Counter, set_cuda_visible_devices, get_ip,
def _init_tokenizer(self, **tokenizer_init_kwargs):
    init_kwargs = dict(enable_lora=bool(self.lora_config), max_num_seqs=self.scheduler_config.max_num_seqs, max_input_length=None, tokenizer_mode=self.model_config.tokenizer_mode, trust_remote_code=self.model_config.trust_remote_code, revision=self.model_config.tokenizer_revision)
    init_kwargs.update(tokenizer_init_kwargs)
    self.tokenizer: TokenizerGroup = TokenizerGroup(self.model_config.tokenizer, **init_kwargs)