import math
from typing import TYPE_CHECKING, Dict, Optional, Set, Type, Union
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from outlines.fsm.guide import CFGGuide, Guide, RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import convert_json_schema_to_str
Compile the FSM that drives the CFG-guided generation.

        Parameters
        ----------
        cfg_str
            A string that represents a grammar
        llm
            The Llama model.
        