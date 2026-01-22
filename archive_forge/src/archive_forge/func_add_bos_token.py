import os
from shutil import copyfile
from typing import Optional, Tuple
from tokenizers import processors
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
from ...utils.versions import require_version
@add_bos_token.setter
def add_bos_token(self, value):
    self._add_bos_token = value
    self.update_post_processor()