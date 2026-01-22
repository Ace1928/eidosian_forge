import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class Qwen2Converter(Converter):

    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, dropout=None, unk_token=None, continuing_subword_prefix='', end_of_word_suffix='', fuse_unk=False, byte_fallback=False))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"), behavior='isolated', invert=False), pre_tokenizers.ByteLevel(add_prefix_space=getattr(self.original_tokenizer, 'add_prefix_space', False), use_regex=False)])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        return tokenizer