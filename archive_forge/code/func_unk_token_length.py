import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
@property
def unk_token_length(self):
    return len(self.sp_model.encode(str(self.unk_token)))