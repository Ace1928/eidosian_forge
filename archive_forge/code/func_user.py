from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
@property
def user(self):
    return f'{self.bos_token}user'