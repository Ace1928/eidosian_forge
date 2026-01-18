import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
@pytest.mark.parametrize('invalid_kwargs', [{'getter': None, 'setter': lambda: None}, {'default': None, 'method': lambda: None, 'getter': lambda: None}, {'setter': lambda: None}, {'default': None, 'method': lambda: None}, {'getter': True}])
def test_underscore_raises_for_invalid(invalid_kwargs):
    invalid_kwargs['force'] = True
    with pytest.raises(ValueError):
        Doc.set_extension('test', **invalid_kwargs)