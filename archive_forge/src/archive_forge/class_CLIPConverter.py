import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class CLIPConverter(Converter):

    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        unk_token = self.original_tokenizer.unk_token
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, dropout=None, continuing_subword_prefix='', end_of_word_suffix='</w>', fuse_unk=False, unk_token=str(unk_token)))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Replace(Regex('\\s+'), ' '), normalizers.Lowercase()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex("'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"), behavior='removed', invert=True), pre_tokenizers.ByteLevel(add_prefix_space=False)])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.RobertaProcessing(sep=(self.original_tokenizer.eos_token, self.original_tokenizer.eos_token_id), cls=(self.original_tokenizer.bos_token, self.original_tokenizer.bos_token_id), add_prefix_space=False, trim_offsets=False)
        return tokenizer