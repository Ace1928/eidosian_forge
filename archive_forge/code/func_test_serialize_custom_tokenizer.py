import pickle
import re
import pytest
from spacy.attrs import ENT_IOB, ENT_TYPE
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import (
from ..util import assert_packed_msg_equal, make_tempdir
def test_serialize_custom_tokenizer(en_vocab, en_tokenizer):
    """Test that custom tokenizer with not all functions defined or empty
    properties can be serialized and deserialized correctly (see #2494,
    #4991)."""
    tokenizer = Tokenizer(en_vocab, suffix_search=en_tokenizer.suffix_search)
    tokenizer_bytes = tokenizer.to_bytes()
    Tokenizer(en_vocab).from_bytes(tokenizer_bytes)
    tokenizer = get_lang_class('en')().tokenizer
    tokenizer.token_match = re.compile('test').match
    assert tokenizer.rules != {}
    assert tokenizer.token_match is not None
    assert tokenizer.url_match is not None
    assert tokenizer.prefix_search is not None
    assert tokenizer.infix_finditer is not None
    tokenizer.from_bytes(tokenizer_bytes)
    assert tokenizer.rules == {}
    assert tokenizer.token_match is None
    assert tokenizer.url_match is None
    assert tokenizer.prefix_search is None
    assert tokenizer.infix_finditer is None
    tokenizer = Tokenizer(en_vocab, rules={'ABC.': [{'ORTH': 'ABC'}, {'ORTH': '.'}]})
    tokenizer.rules = {}
    tokenizer_bytes = tokenizer.to_bytes()
    tokenizer_reloaded = Tokenizer(en_vocab).from_bytes(tokenizer_bytes)
    assert tokenizer_reloaded.rules == {}