import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
def append_token_id(self, token_id: int, logprobs: Dict[int, float]) -> None:
    assert token_id in logprobs
    self._append_tokens_to_blocks([token_id])
    self.output_logprobs.append(logprobs)
    self.data.append_token_id(token_id, logprobs[token_id])