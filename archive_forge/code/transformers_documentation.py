from collections import defaultdict
from typing import DefaultDict, List, Optional, Type, Union
import torch
from pydantic import BaseModel
from transformers import Pipeline, PreTrainedTokenizerBase
from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import adapt_tokenizer, convert_json_schema_to_str
Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A schema that encodes the structure we want the model to generate.
        tokenizer_or_pipe
            The tokenizer of the model, or the pipeline object.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[
 ]?"`
        