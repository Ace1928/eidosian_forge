import json
import os
import re
import unicodedata
from typing import Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging
Converts a sequence of tokens (string) in a single string.