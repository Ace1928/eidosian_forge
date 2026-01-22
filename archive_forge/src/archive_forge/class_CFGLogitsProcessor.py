import math
from typing import TYPE_CHECKING, Dict, Optional, Set, Type, Union
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from outlines.fsm.guide import CFGGuide, Guide, RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import convert_json_schema_to_str
class CFGLogitsProcessor(LogitsProcessor):
    """Bias LlamaCpp generation based on a context-free grammar.

    Attributes
    ----------
    llm
        The Llama model.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, cfg_str: str, llm: 'Llama'):
        """Compile the FSM that drives the CFG-guided generation.

        Parameters
        ----------
        cfg_str
            A string that represents a grammar
        llm
            The Llama model.
        """
        tokenizer = LlamaCppTokenizer(model=llm)
        fsm = CFGGuide(cfg_string=cfg_str, tokenizer=tokenizer)
        super().__init__(tokenizer=tokenizer, fsm=fsm)