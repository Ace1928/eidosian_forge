import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams
import torch
def to_sampling_params(self):
    echo_without_generation = self.echo and self.max_tokens == 0
    logits_processors = None
    if self.logit_bias:

        def logit_bias_logits_processor(token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
            for token_id, bias in self.logit_bias.items():
                bias = min(100, max(-100, bias))
                logits[int(token_id)] += bias
            return logits
        logits_processors = [logit_bias_logits_processor]
    return SamplingParams(n=self.n, best_of=self.best_of, presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty, repetition_penalty=self.repetition_penalty, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, min_p=self.min_p, seed=self.seed, stop=self.stop, stop_token_ids=self.stop_token_ids, ignore_eos=self.ignore_eos, max_tokens=self.max_tokens if not echo_without_generation else 1, logprobs=self.logprobs, use_beam_search=self.use_beam_search, early_stopping=self.early_stopping, prompt_logprobs=self.logprobs if self.echo else None, skip_special_tokens=self.skip_special_tokens, spaces_between_special_tokens=self.spaces_between_special_tokens, include_stop_str_in_output=self.include_stop_str_in_output, length_penalty=self.length_penalty, logits_processors=logits_processors)