import json
import os
from typing import Optional, Tuple
import regex as re
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
Converts a sequence of tokens (string) in a single string.