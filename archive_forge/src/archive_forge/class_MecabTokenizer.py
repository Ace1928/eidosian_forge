import collections
import copy
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import is_sentencepiece_available, is_sudachi_projection_available, logging
class MecabTokenizer:
    """Runs basic tokenization with MeCab morphological parser."""

    def __init__(self, do_lower_case=False, never_split=None, normalize_text=True, mecab_dic: Optional[str]='ipadic', mecab_option: Optional[str]=None):
        """
        Constructs a MecabTokenizer.

        Args:
            **do_lower_case**: (*optional*) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (*optional*) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
            **normalize_text**: (*optional*) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **mecab_dic**: (*optional*) string (default "ipadic")
                Name of dictionary to be used for MeCab initialization. If you are using a system-installed dictionary,
                set this option to `None` and modify *mecab_option*.
            **mecab_option**: (*optional*) string
                String passed to MeCab constructor.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text
        try:
            import fugashi
        except ModuleNotFoundError as error:
            raise error.__class__('You need to install fugashi to use MecabTokenizer. See https://pypi.org/project/fugashi/ for installation.')
        mecab_option = mecab_option or ''
        if mecab_dic is not None:
            if mecab_dic == 'ipadic':
                try:
                    import ipadic
                except ModuleNotFoundError as error:
                    raise error.__class__('The ipadic dictionary is not installed. See https://github.com/polm/ipadic-py for installation.')
                dic_dir = ipadic.DICDIR
            elif mecab_dic == 'unidic_lite':
                try:
                    import unidic_lite
                except ModuleNotFoundError as error:
                    raise error.__class__('The unidic_lite dictionary is not installed. See https://github.com/polm/unidic-lite for installation.')
                dic_dir = unidic_lite.DICDIR
            elif mecab_dic == 'unidic':
                try:
                    import unidic
                except ModuleNotFoundError as error:
                    raise error.__class__('The unidic dictionary is not installed. See https://github.com/polm/unidic-py for installation.')
                dic_dir = unidic.DICDIR
                if not os.path.isdir(dic_dir):
                    raise RuntimeError('The unidic dictionary itself is not found. See https://github.com/polm/unidic-py for installation.')
            else:
                raise ValueError('Invalid mecab_dic is specified.')
            mecabrc = os.path.join(dic_dir, 'mecabrc')
            mecab_option = f'-d "{dic_dir}" -r "{mecabrc}" ' + mecab_option
        self.mecab = fugashi.GenericTagger(mecab_option)

    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize('NFKC', text)
        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = []
        for word in self.mecab(text):
            token = word.surface
            if self.do_lower_case and token not in never_split:
                token = token.lower()
            tokens.append(token)
        return tokens