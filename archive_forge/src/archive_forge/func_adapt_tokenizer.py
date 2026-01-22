import json
import math
from collections import defaultdict
from typing import Union, DefaultDict, Dict, List, Optional
import torch
from pydantic import BaseModel
from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema
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