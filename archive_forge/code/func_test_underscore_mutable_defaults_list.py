import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
def test_underscore_mutable_defaults_list(en_vocab):
    """Test that mutable default arguments are handled correctly (see #2581)."""
    Doc.set_extension('mutable', default=[])
    doc1 = Doc(en_vocab, words=['one'])
    doc2 = Doc(en_vocab, words=['two'])
    doc1._.mutable.append('foo')
    assert len(doc1._.mutable) == 1
    assert doc1._.mutable[0] == 'foo'
    assert len(doc2._.mutable) == 0
    doc1._.mutable = ['bar', 'baz']
    doc1._.mutable.append('foo')
    assert len(doc1._.mutable) == 3
    assert len(doc2._.mutable) == 0