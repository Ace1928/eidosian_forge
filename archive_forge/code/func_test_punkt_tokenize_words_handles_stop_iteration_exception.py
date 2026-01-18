from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_punkt_tokenize_words_handles_stop_iteration_exception(self):
    obj = punkt.PunktBaseClass()

    class TestPunktTokenizeWordsMock:

        def word_tokenize(self, s):
            return iter([])
    obj._lang_vars = TestPunktTokenizeWordsMock()
    list(obj._tokenize_words('test'))