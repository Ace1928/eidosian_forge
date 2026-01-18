import os
from shutil import copyfile
from typing import List, Optional, Tuple
from tokenizers import processors
from ...tokenization_utils import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code].