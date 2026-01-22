from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
from typing import Dict, Iterable, List, Tuple, Union, Type, Callable
from utils.log import quick_log
from fastapi import HTTPException
from pydantic import BaseModel, Field
from routes import state_cache
import global_var
class MusicAbcRWKV(AbstractRWKV):

    def __init__(self, model, pipeline):
        super().__init__(model, pipeline)
        self.EOS_ID = 3
        self.max_tokens_per_generation = 500
        self.temperature = 1
        self.top_p = 0.8
        self.top_k = 8
        self.rwkv_type = RWKVType.Music

    def adjust_occurrence(self, occurrence: Dict, token: int):
        pass

    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        pass

    def fix_tokens(self, tokens) -> List[int]:
        return tokens

    def run_rnn(self, _tokens: List[str], newline_adj: int=0) -> Tuple[List[float], int]:
        tokens = [int(x) for x in _tokens]
        token_len = len(tokens)
        self.model_tokens += tokens
        out, self.model_state = self.model.forward(tokens, self.model_state)
        return (out, token_len)

    def delta_postprocess(self, delta: str) -> str:
        return delta