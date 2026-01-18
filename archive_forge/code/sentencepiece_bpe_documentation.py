from typing import Dict, Iterator, List, Optional, Tuple, Union
from tokenizers import AddedToken, Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from .base_tokenizer import BaseTokenizer
Train the model using the given iterator