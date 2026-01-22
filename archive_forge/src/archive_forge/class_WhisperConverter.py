import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class WhisperConverter(Converter):

    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, dropout=None, continuing_subword_prefix='', end_of_word_suffix='', fuse_unk=False))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        prefix_token_ids = self.original_tokenizer.prefix_tokens
        prefixes = self.original_tokenizer.convert_ids_to_tokens(prefix_token_ids)
        eos = self.original_tokenizer.eos_token
        eos_token_id = self.original_tokenizer.eos_token_id
        prefix_template = ' '.join([f'{token}:0' for token in prefixes])
        tokenizer.post_processor = processors.TemplateProcessing(single=f'{prefix_template} $A:0 {eos}:0', pair=f'{prefix_template} $A:0 $B:1 {eos}:1', special_tokens=[(eos, eos_token_id), *zip(prefixes, prefix_token_ids)])
        return tokenizer