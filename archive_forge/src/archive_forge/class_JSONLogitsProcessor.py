import json
import math
from collections import defaultdict
from typing import Union, DefaultDict, Dict, List, Optional
import torch
from pydantic import BaseModel
from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema
class JSONLogitsProcessor(RegexLogitsProcessor):

    def __init__(self, schema: Union[str, Dict, BaseModel], tokenizer, whitespace_pattern: Optional[str]=None):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate
        tokenizer
            The model's tokenizer
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
            Example: allow only a single space or newline with `whitespace_pattern=r"[
 ]?"`
        """
        if isinstance(schema, type(BaseModel)):
            schema_str = json.dumps(schema.model_json_schema())
        elif isinstance(schema, Dict):
            schema_str = json.dumps(schema)
        elif isinstance(schema, str):
            schema_str = schema
        else:
            raise ValueError(f'Cannot parse schema {schema}. The schema must be either ' + 'a Pydantic object, a dictionary or a string that contains the JSON ' + 'Schema specification')
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string, tokenizer)