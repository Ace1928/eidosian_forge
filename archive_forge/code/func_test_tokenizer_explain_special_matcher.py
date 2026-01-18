import re
import string
import hypothesis
import hypothesis.strategies
import pytest
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import get_lang_class
def test_tokenizer_explain_special_matcher(en_vocab):
    suffix_re = re.compile('[\\.]$')
    infix_re = re.compile('[/]')
    rules = {'a.': [{'ORTH': 'a.'}]}
    tokenizer = Tokenizer(en_vocab, rules=rules, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer)
    tokens = [t.text for t in tokenizer('a/a.')]
    explain_tokens = [t[1] for t in tokenizer.explain('a/a.')]
    assert tokens == explain_tokens