import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class SplinterConverter(Converter):

    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))
        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, 'basic_tokenizer'):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case
        tokenizer.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=tokenize_chinese_chars, strip_accents=strip_accents, lowercase=do_lower_case)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        question = str(self.original_tokenizer.question_token)
        dot = '.'
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id
        question_token_id = self.original_tokenizer.question_token_id
        dot_token_id = self.original_tokenizer.convert_tokens_to_ids('.')
        if self.original_tokenizer.padding_side == 'right':
            pair = f'{cls}:0 $A:0 {question} {dot} {sep}:0 $B:1 {sep}:1'
        else:
            pair = f'{cls}:0 $A:0 {sep}:0 $B:1 {question} {dot} {sep}:1'
        tokenizer.post_processor = processors.TemplateProcessing(single=f'{cls}:0 $A:0 {sep}:0', pair=pair, special_tokens=[(cls, cls_token_id), (sep, sep_token_id), (question, question_token_id), (dot, dot_token_id)])
        tokenizer.decoder = decoders.WordPiece(prefix='##')
        return tokenizer