import os
import re
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_torch_available, logging

        This chat template formats messages like an instant messenger chat log, with "User:" and "Bot:" strings
        preceding messages. BOS tokens are added between all messages.
        