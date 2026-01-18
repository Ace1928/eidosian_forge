import re
import numpy
import pytest
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import (
from spacy.vocab import Vocab
def test_tokenizer_special_cases_with_affixes_preserve_spacy():
    tokenizer = English().tokenizer
    tokenizer.rules = {}
    text = "''a'' "
    tokenizer.add_special_case("''", [{'ORTH': "''"}])
    assert tokenizer(text).text == text
    tokenizer.add_special_case('ab', [{'ORTH': 'a'}, {'ORTH': 'b'}])
    text = "ab ab ab ''ab ab'' ab'' ''ab"
    assert tokenizer(text).text == text