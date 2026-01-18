from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_punkt_debug_decisions_custom_end(self):

    class ExtLangVars(punkt.PunktLanguageVars):
        sent_end_chars = ('.', '?', '!', '^')
    self.punkt_debug_decisions('Subject: Some subject^ Attachments: Some attachments', n_sents=2, n_splits=1, lang_vars=ExtLangVars())