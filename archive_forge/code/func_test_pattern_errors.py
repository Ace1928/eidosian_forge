import pytest
from spacy.errors import MatchPatternError
from spacy.matcher import Matcher
from spacy.schemas import validate_token_pattern
def test_pattern_errors(en_vocab):
    matcher = Matcher(en_vocab)
    matcher.add('TEST1', [[{'text': {'regex': 'regex'}}]])
    with pytest.raises(MatchPatternError):
        matcher.add('TEST2', [[{'TEXT': {'XX': 'xx'}}]])