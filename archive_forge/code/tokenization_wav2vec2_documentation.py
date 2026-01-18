import json
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...utils import (

        special _decode function is needed for Wav2Vec2Tokenizer because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        