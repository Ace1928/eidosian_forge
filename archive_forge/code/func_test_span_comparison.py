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
def test_span_comparison(doc):
    assert Span(doc, 0, 3) == Span(doc, 0, 3)
    assert Span(doc, 0, 3, 'LABEL') == Span(doc, 0, 3, 'LABEL')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') == Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3) != Span(doc, 0, 3, 'LABEL')
    assert Span(doc, 0, 3) != Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL') != Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3) <= Span(doc, 0, 3) and Span(doc, 0, 3) >= Span(doc, 0, 3)
    assert Span(doc, 0, 3, 'LABEL') <= Span(doc, 0, 3, 'LABEL') and Span(doc, 0, 3, 'LABEL') >= Span(doc, 0, 3, 'LABEL')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') <= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') >= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3) < Span(doc, 0, 3, '', kb_id='KB_ID') < Span(doc, 0, 3, 'LABEL') < Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3) <= Span(doc, 0, 3, '', kb_id='KB_ID') <= Span(doc, 0, 3, 'LABEL') <= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') > Span(doc, 0, 3, 'LABEL') > Span(doc, 0, 3, '', kb_id='KB_ID') > Span(doc, 0, 3)
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') >= Span(doc, 0, 3, 'LABEL') >= Span(doc, 0, 3, '', kb_id='KB_ID') >= Span(doc, 0, 3)
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') < Span(doc, 0, 4, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') < Span(doc, 0, 4)
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') <= Span(doc, 0, 4)
    assert Span(doc, 0, 4) > Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 4) >= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') != Span(doc, 1, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') < Span(doc, 1, 3)
    assert Span(doc, 0, 3, 'LABEL', kb_id='KB_ID') <= Span(doc, 1, 3)
    assert Span(doc, 1, 3) > Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 1, 3) >= Span(doc, 0, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 4, 'LABEL', kb_id='KB_ID') != Span(doc, 1, 3, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 0, 4, 'LABEL', kb_id='KB_ID') < Span(doc, 1, 3)
    assert Span(doc, 0, 4, 'LABEL', kb_id='KB_ID') <= Span(doc, 1, 3)
    assert Span(doc, 1, 3) > Span(doc, 0, 4, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 1, 3) >= Span(doc, 0, 4, 'LABEL', kb_id='KB_ID')
    assert Span(doc, 1, 3, span_id='AAA') < Span(doc, 1, 3, span_id='BBB')