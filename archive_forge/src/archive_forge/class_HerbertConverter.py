import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class HerbertConverter(Converter):

    def converted(self) -> Tokenizer:
        tokenizer_info_str = '#version:'
        token_suffix = '</w>'
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        if tokenizer_info_str in merges[0][0]:
            merges = merges[1:]
        tokenizer = Tokenizer(BPE(vocab, merges, dropout=None, unk_token=self.original_tokenizer.unk_token, end_of_word_suffix=token_suffix))
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder(suffix=token_suffix)
        tokenizer.post_processor = processors.BertProcessing(sep=(self.original_tokenizer.sep_token, self.original_tokenizer.sep_token_id), cls=(self.original_tokenizer.cls_token, self.original_tokenizer.cls_token_id))
        return tokenizer