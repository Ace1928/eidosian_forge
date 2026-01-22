import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class LlamaConverter(SpmConverter):
    handle_byte_fallback = True

    def vocab(self, proto):
        vocab = [('<unk>', 0.0), ('<s>', 0.0), ('</s>', 0.0)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 0
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        sequence = [decoders.Replace('▁', ' '), decoders.ByteFallback(), decoders.Fuse()]
        if add_prefix_space:
            sequence += [decoders.Strip(content=' ', left=1)]
        return decoders.Sequence(sequence)

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        if model_type == 1:
            import tokenizers
            if version.parse(tokenizers.__version__) < version.parse('0.14.0'):
                tokenizer = Tokenizer(Unigram(vocab_scores, 0))
            else:
                tokenizer = Tokenizer(Unigram(vocab_scores, 0, byte_fallback=True))
        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True))
            tokenizer.add_special_tokens([AddedToken('<unk>', normalized=False, special=True), AddedToken('<s>', normalized=False, special=True), AddedToken('</s>', normalized=False, special=True)])
        else:
            raise Exception("You're trying to run a `Unigram` model but you're file was trained with a different algorithm")
        return tokenizer

    def normalizer(self, proto):
        sequence = []
        if hasattr(self.original_tokenizer, 'add_prefix_space'):
            if self.original_tokenizer.add_prefix_space:
                sequence += [normalizers.Prepend(prepend='▁')]
        sequence += [normalizers.Replace(pattern=' ', content='▁')]
        return normalizers.Sequence(sequence)

    def pre_tokenizer(self, replacement, add_prefix_space):
        return None

    def post_processor(self):
        return None