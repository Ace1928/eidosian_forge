import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

        A very simple chat template that just adds whitespace between messages.
        