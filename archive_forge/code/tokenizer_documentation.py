from typing import List, Optional, Tuple, Union
from transformers import (AutoTokenizer, PreTrainedTokenizer,
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils import make_async, LRUCache
from vllm.transformers_utils.tokenizers import *
A group of tokenizers that can be used for LoRA adapters.