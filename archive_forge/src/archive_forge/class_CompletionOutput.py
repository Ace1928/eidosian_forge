from typing import List, Optional
import time
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SequenceGroup,
from vllm.lora.request import LoRARequest
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(self, index: int, text: str, token_ids: List[int], cumulative_logprob: float, logprobs: Optional[SampleLogprobs], finish_reason: Optional[str]=None, lora_request: Optional[LoRARequest]=None) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.lora_request = lora_request

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return f'CompletionOutput(index={self.index}, text={self.text!r}, token_ids={self.token_ids}, cumulative_logprob={self.cumulative_logprob}, logprobs={self.logprobs}, finish_reason={self.finish_reason})'