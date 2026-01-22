import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams
import torch
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    seed: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    early_stopping: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.0
    min_p: Optional[float] = 0.0
    include_stop_str_in_output: Optional[bool] = False
    length_penalty: Optional[float] = 1.0
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None

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

    @model_validator(mode='before')
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum(['guided_json' in data and data['guided_json'] is not None, 'guided_regex' in data and data['guided_regex'] is not None, 'guided_choice' in data and data['guided_choice'] is not None])
        if guide_count > 1:
            raise ValueError("You can only use one kind of guided decoding ('guided_json', 'guided_regex' or 'guided_choice').")
        return data