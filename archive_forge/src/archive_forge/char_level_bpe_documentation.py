from typing import Dict, Iterator, List, Optional, Tuple, Union
from .. import AddedToken, Tokenizer, decoders, pre_tokenizers, trainers
from ..models import BPE
from ..normalizers import BertNormalizer, Lowercase, Sequence, unicode_normalizer_from_str
from .base_tokenizer import BaseTokenizer
Train the model using the given iterator