import os
from shutil import copyfile
from typing import List, Optional, Tuple
from tokenizers import processors
from ...tokenization_utils import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
Used by translation pipeline, to prepare inputs for the generate function