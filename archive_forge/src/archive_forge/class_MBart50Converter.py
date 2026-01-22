import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class MBart50Converter(SpmConverter):

    def vocab(self, proto):
        vocab = [('<s>', 0.0), ('<pad>', 0.0), ('</s>', 0.0), ('<unk>', 0.0)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [('ar_AR', 0.0), ('cs_CZ', 0.0), ('de_DE', 0.0), ('en_XX', 0.0), ('es_XX', 0.0), ('et_EE', 0.0), ('fi_FI', 0.0), ('fr_XX', 0.0), ('gu_IN', 0.0), ('hi_IN', 0.0), ('it_IT', 0.0), ('ja_XX', 0.0), ('kk_KZ', 0.0), ('ko_KR', 0.0), ('lt_LT', 0.0), ('lv_LV', 0.0), ('my_MM', 0.0), ('ne_NP', 0.0), ('nl_XX', 0.0), ('ro_RO', 0.0), ('ru_RU', 0.0), ('si_LK', 0.0), ('tr_TR', 0.0), ('vi_VN', 0.0), ('zh_CN', 0.0), ('af_ZA', 0.0), ('az_AZ', 0.0), ('bn_IN', 0.0), ('fa_IR', 0.0), ('he_IL', 0.0), ('hr_HR', 0.0), ('id_ID', 0.0), ('ka_GE', 0.0), ('km_KH', 0.0), ('mk_MK', 0.0), ('ml_IN', 0.0), ('mn_MN', 0.0), ('mr_IN', 0.0), ('pl_PL', 0.0), ('ps_AF', 0.0), ('pt_XX', 0.0), ('sv_SE', 0.0), ('sw_KE', 0.0), ('ta_IN', 0.0), ('te_IN', 0.0), ('th_TH', 0.0), ('tl_XX', 0.0), ('uk_UA', 0.0), ('ur_PK', 0.0), ('xh_ZA', 0.0), ('gl_ES', 0.0), ('sl_SI', 0.0)]
        vocab += [('<mask>', 0.0)]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(single='en_XX $A </s>', pair='en_XX $A $B </s>', special_tokens=[('en_XX', self.original_tokenizer.convert_tokens_to_ids('en_XX')), ('</s>', self.original_tokenizer.convert_tokens_to_ids('</s>'))])