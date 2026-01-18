import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
def test_span_underscore_getter_setter():
    span = Mock(doc=Mock(), start=0, end=2)
    Underscore.span_extensions['hello'] = (None, None, lambda s: (s.start, 'hi'), lambda s, value: setattr(s, 'start', value))
    span._ = Underscore(Underscore.span_extensions, span, start=span.start, end=span.end)
    assert span._.hello == (0, 'hi')
    span._.hello = 1
    assert span._.hello == (1, 'hi')