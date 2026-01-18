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
@pytest.mark.parametrize('start,end,expected_sentence', [(0, 14, 'This is'), (1, 4, 'This is'), (0, 2, 'This is'), (0, 1, 'This is'), (10, 14, 'And a'), (12, 14, 'third.'), (1, 1, 'This is')])
def test_spans_span_sent_user_hooks(doc, start, end, expected_sentence):

    def user_hook(doc):
        return [doc[ii:ii + 2] for ii in range(0, len(doc), 2)]
    doc.user_hooks['sents'] = user_hook
    assert doc[start:end].sent.text == expected_sentence
    doc.user_span_hooks['sent'] = lambda x: x
    assert doc[start:end].sent == doc[start:end]