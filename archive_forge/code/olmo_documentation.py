from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (
from vllm.sequence import SamplerOutput
from hf_olmo import OLMoConfig

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        