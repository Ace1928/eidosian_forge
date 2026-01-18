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
@pytest.mark.parametrize('start,end,expected_sentences,expected_sentences_with_hook', [(0, 14, 3, 7), (3, 6, 2, 2), (0, 4, 1, 2), (0, 3, 1, 2), (9, 14, 2, 3), (10, 14, 1, 2), (11, 14, 1, 2), (0, 0, 1, 1)])
def test_span_sents(doc, start, end, expected_sentences, expected_sentences_with_hook):
    assert len(list(doc[start:end].sents)) == expected_sentences

    def user_hook(doc):
        return [doc[ii:ii + 2] for ii in range(0, len(doc), 2)]
    doc.user_hooks['sents'] = user_hook
    assert len(list(doc[start:end].sents)) == expected_sentences_with_hook
    doc.user_span_hooks['sents'] = lambda x: [x]
    assert list(doc[start:end].sents)[0] == doc[start:end]
    assert len(list(doc[start:end].sents)) == 1