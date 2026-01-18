import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import get_current_ops
from spacy.attrs import LENGTH, ORTH
from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab
from .test_underscore import clean_underscore  # noqa: F401
@pytest.mark.usefixtures('clean_underscore')
def test_span_as_doc_user_data(doc):
    """Test that the user_data can be preserved (but not by default)."""
    my_key = 'my_info'
    my_value = 342
    doc.user_data[my_key] = my_value
    Token.set_extension('is_x', default=False)
    doc[7]._.is_x = True
    span = doc[4:10]
    span_doc_with = span.as_doc(copy_user_data=True)
    span_doc_without = span.as_doc()
    assert doc.user_data.get(my_key, None) is my_value
    assert span_doc_with.user_data.get(my_key, None) is my_value
    assert span_doc_without.user_data.get(my_key, None) is None
    for i in range(len(span_doc_with)):
        if i != 3:
            assert span_doc_with[i]._.is_x is False
        else:
            assert span_doc_with[i]._.is_x is True
    assert not any([t._.is_x for t in span_doc_without])