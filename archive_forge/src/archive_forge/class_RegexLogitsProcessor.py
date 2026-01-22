import json
import math
from collections import defaultdict
from typing import Union, DefaultDict, Dict, List, Optional
import torch
from pydantic import BaseModel
from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema
class RegexLogitsProcessor:

    def __init__(self, regex_string: str, tokenizer):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            The model's tokenizer

        """
        tokenizer = self.adapt_tokenizer(tokenizer)
        fsm = RegexFSM(regex_string, tokenizer)
        self.fsm = fsm

    def init_state(self):
        """Initialize the FSM states."""
        self.fsm_state: DefaultDict[int, int] = defaultdict(int)

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""
        seq_id = hash(tuple(input_ids))
        if len(input_ids) == 0:
            self.init_state()
        else:
            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            self.fsm_state[seq_id] = self.fsm.next_state(self.fsm_state[last_seq_id], last_token)
        allowed_tokens = self.fsm.allowed_token_ids(self.fsm_state[seq_id])
        mask = torch.full((scores.shape[-1],), -math.inf, device=scores.device)
        mask[allowed_tokens] = 0
        scores.add_(mask)
        return scores

    def adapt_tokenizer(self, tokenizer):
        """Adapt vLLM's tokenizer to use to compile the FSM.

        The API of Outlines tokenizers is slightly different to that of
        `transformers`. In addition we need to handle the missing spaces to
        Llama's tokenizer to be able to compile FSMs for this model.

        """
        tokenizer.vocabulary = tokenizer.get_vocab()
        tokenizer.special_tokens = set(tokenizer.all_special_tokens)

        def convert_token_to_string(token: str) -> str:
            from transformers.file_utils import SPIECE_UNDERLINE
            string = tokenizer.convert_tokens_to_string([token])
            if token.startswith(SPIECE_UNDERLINE) or token == '<0x20>':
                return ' ' + string
            return string
        tokenizer.convert_token_to_string = convert_token_to_string
        return tokenizer