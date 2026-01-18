import json
import math
from collections import defaultdict
from typing import Union, DefaultDict, Dict, List, Optional
import torch
from pydantic import BaseModel
from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema
Compile the FSM that drives the JSON-guided generation.

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
        