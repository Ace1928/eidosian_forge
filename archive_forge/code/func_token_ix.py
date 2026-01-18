import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
@property
def token_ix(self):
    return self.tokens[0] if len(self.tokens) > 0 else None