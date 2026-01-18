import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
@pytest.mark.parametrize('obj', [Doc, Span, Token])
def test_underscore_raises_for_dup(obj):
    obj.set_extension('test', default=None)
    with pytest.raises(ValueError):
        obj.set_extension('test', default=None)