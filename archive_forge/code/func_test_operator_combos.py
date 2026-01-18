import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
def test_operator_combos(en_vocab):
    cases = [('aaab', 'a a a b', True), ('aaab', 'a+ b', True), ('aaab', 'a+ a+ b', True), ('aaab', 'a+ a+ a b', True), ('aaab', 'a+ a+ a+ b', True), ('aaab', 'a+ a a b', True), ('aaab', 'a+ a a', True), ('aaab', 'a+', True), ('aaa', 'a+ b', False), ('aaa', 'a+ a+ b', False), ('aaa', 'a+ a+ a+ b', False), ('aaa', 'a+ a b', False), ('aaa', 'a+ a a b', False), ('aaab', 'a+ a a', True), ('aaab', 'a+', True), ('aaab', 'a+ a b', True)]
    for string, pattern_str, result in cases:
        matcher = Matcher(en_vocab)
        doc = Doc(matcher.vocab, words=list(string))
        pattern = []
        for part in pattern_str.split():
            if part.endswith('+'):
                pattern.append({'ORTH': part[0], 'OP': '+'})
            else:
                pattern.append({'ORTH': part})
        matcher.add('PATTERN', [pattern])
        matches = matcher(doc)
        if result:
            assert matches, (string, pattern_str)
        else:
            assert not matches, (string, pattern_str)