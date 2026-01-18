import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
def test_underscore_mutable_defaults_dict(en_vocab):
    """Test that mutable default arguments are handled correctly (see #2581)."""
    Token.set_extension('mutable', default={})
    token1 = Doc(en_vocab, words=['one'])[0]
    token2 = Doc(en_vocab, words=['two'])[0]
    token1._.mutable['foo'] = 'bar'
    assert len(token1._.mutable) == 1
    assert token1._.mutable['foo'] == 'bar'
    assert len(token2._.mutable) == 0
    token1._.mutable['foo'] = 'baz'
    assert len(token1._.mutable) == 1
    assert token1._.mutable['foo'] == 'baz'
    token1._.mutable['x'] = []
    token1._.mutable['x'].append('y')
    assert len(token1._.mutable) == 2
    assert token1._.mutable['x'] == ['y']
    assert len(token2._.mutable) == 0