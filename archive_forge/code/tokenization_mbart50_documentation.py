import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from ...utils import logging
Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos].