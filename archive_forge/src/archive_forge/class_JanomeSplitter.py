import os
import re
import sys
from typing import Any, Dict, List
from sphinx.errors import ExtensionError, SphinxError
from sphinx.search import SearchLanguage
from sphinx.util import import_object
class JanomeSplitter(BaseSplitter):

    def __init__(self, options: Dict) -> None:
        super().__init__(options)
        self.user_dict = options.get('user_dic')
        self.user_dict_enc = options.get('user_dic_enc', 'utf8')
        self.init_tokenizer()

    def init_tokenizer(self) -> None:
        if not janome_module:
            raise RuntimeError('Janome is not available')
        self.tokenizer = janome.tokenizer.Tokenizer(udic=self.user_dict, udic_enc=self.user_dict_enc)

    def split(self, input: str) -> List[str]:
        result = ' '.join((token.surface for token in self.tokenizer.tokenize(input)))
        return result.split(' ')