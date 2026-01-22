import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class DebertaConverter(Converter):

    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, dropout=None, continuing_subword_prefix='', end_of_word_suffix='', fuse_unk=False))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.TemplateProcessing(single='[CLS]:0 $A:0 [SEP]:0', pair='[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1', special_tokens=[('[CLS]', self.original_tokenizer.convert_tokens_to_ids('[CLS]')), ('[SEP]', self.original_tokenizer.convert_tokens_to_ids('[SEP]'))])
        return tokenizer