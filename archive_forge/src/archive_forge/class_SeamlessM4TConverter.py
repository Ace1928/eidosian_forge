import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class SeamlessM4TConverter(SpmConverter):

    def vocab(self, proto):
        vocab = [('<pad>', 0.0), ('<unk>', 0.0), ('<s>', 0.0), ('</s>', 0.0)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        return self.original_tokenizer.unk_token_id

    def post_processor(self):
        return processors.TemplateProcessing(single='__eng__ $A </s>', pair='__eng__ $A $B </s>', special_tokens=[('__eng__', self.original_tokenizer.convert_tokens_to_ids('__eng__')), ('</s>', self.original_tokenizer.convert_tokens_to_ids('</s>'))])