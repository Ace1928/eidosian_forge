import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class AlbertConverter(SpmConverter):

    def vocab(self, proto):
        return [(piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100) for piece in proto.pieces]

    def normalizer(self, proto):
        list_normalizers = [normalizers.Replace('``', '"'), normalizers.Replace("''", '"')]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        list_normalizers.append(normalizers.Replace(Regex(' {2,}'), ' '))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(single='[CLS]:0 $A:0 [SEP]:0', pair='[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1', special_tokens=[('[CLS]', self.original_tokenizer.convert_tokens_to_ids('[CLS]')), ('[SEP]', self.original_tokenizer.convert_tokens_to_ids('[SEP]'))])