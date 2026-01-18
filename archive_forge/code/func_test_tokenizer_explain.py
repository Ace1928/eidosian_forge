import re
import string
import hypothesis
import hypothesis.strategies
import pytest
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import get_lang_class
@pytest.mark.parametrize('lang', LANGUAGES)
def test_tokenizer_explain(lang):
    tokenizer = get_lang_class(lang)().tokenizer
    examples = pytest.importorskip(f'spacy.lang.{lang}.examples')
    for sentence in examples.sentences:
        tokens = [t.text for t in tokenizer(sentence) if not t.is_space]
        debug_tokens = [t[1] for t in tokenizer.explain(sentence)]
        assert tokens == debug_tokens