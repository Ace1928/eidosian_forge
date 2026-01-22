from collections import defaultdict
from typing import DefaultDict, List, Optional, Type, Union
import torch
from pydantic import BaseModel
from transformers import Pipeline, PreTrainedTokenizerBase
from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import adapt_tokenizer, convert_json_schema_to_str
class JSONPrefixAllowedTokens(RegexPrefixAllowedTokens):
    """Bias transformers generation based on a JSON schema.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, schema: Union[dict, Type[BaseModel], str], tokenizer_or_pipe: Union[PreTrainedTokenizerBase, Pipeline], whitespace_pattern: Optional[str]=None):
        """Compile the FSM that drives the JSON-guided generation.

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
        """
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string=regex_string, tokenizer_or_pipe=tokenizer_or_pipe)