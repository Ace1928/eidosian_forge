import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece as spm
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import (
from ...tokenization_utils_base import AddedToken
from ...utils import PaddingStrategy, logging
Reset the special tokens to the target lang setting.
        Prefix=[eos, tgt_lang_code] and suffix=[eos].
        