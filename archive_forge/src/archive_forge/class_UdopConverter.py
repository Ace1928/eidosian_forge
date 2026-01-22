import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class UdopConverter(SpmConverter):

    def post_processor(self):
        return processors.TemplateProcessing(single=['$A', '</s>'], pair=['$A', '</s>', '$B', '</s>'], special_tokens=[('</s>', self.original_tokenizer.convert_tokens_to_ids('</s>'))])