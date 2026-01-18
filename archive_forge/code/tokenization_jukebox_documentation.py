import json
import os
import re
import unicodedata
from json.encoder import INFINITY
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import regex
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_flax_available, is_tf_available, is_torch_available, logging
from ...utils.generic import _is_jax, _is_numpy

        Converts an index (integer) in a token (str) using the vocab.

        Args:
            artists_index (`int`):
                Index of the artist in its corresponding dictionary.
            genres_index (`Union[List[int], int]`):
               Index of the genre in its corresponding dictionary.
            lyric_index (`List[int]`):
                List of character indices, which each correspond to a character.
        