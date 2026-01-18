import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
@pytest.mark.parametrize('valid_kwargs', [{'getter': lambda: None}, {'getter': lambda: None, 'setter': lambda: None}, {'default': 'hello'}, {'default': None}, {'method': lambda: None}])
def test_underscore_accepts_valid(valid_kwargs):
    valid_kwargs['force'] = True
    Doc.set_extension('test', **valid_kwargs)