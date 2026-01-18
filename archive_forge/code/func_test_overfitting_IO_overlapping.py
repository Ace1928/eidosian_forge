import numpy
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from thinc.api import NumpyOps, Ragged, get_current_ops
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.tokens import SpanGroup
from spacy.tokens._dict_proxies import SpanGroups
from spacy.training import Example
from spacy.util import fix_random_seed, make_tempdir, registry
def test_overfitting_IO_overlapping():
    fix_random_seed(0)
    nlp = English()
    spancat = nlp.add_pipe('spancat', config={'spans_key': SPAN_KEY})
    train_examples = make_examples(nlp, data=TRAIN_DATA_OVERLAPPING)
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    assert spancat.model.get_dim('nO') == 3
    assert set(spancat.labels) == {'PERSON', 'LOC', 'DOUBLE_LOC'}
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses['spancat'] < 0.01
    test_text = 'I like London and Berlin'
    doc = nlp(test_text)
    spans = doc.spans[SPAN_KEY]
    assert len(spans) == 3
    assert len(spans.attrs['scores']) == 3
    assert min(spans.attrs['scores']) > 0.9
    assert set([span.text for span in spans]) == {'London', 'Berlin', 'London and Berlin'}
    assert set([span.label_ for span in spans]) == {'LOC', 'DOUBLE_LOC'}
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        spans2 = doc2.spans[SPAN_KEY]
        assert len(spans2) == 3
        assert len(spans2.attrs['scores']) == 3
        assert min(spans2.attrs['scores']) > 0.9
        assert set([span.text for span in spans2]) == {'London', 'Berlin', 'London and Berlin'}
        assert set([span.label_ for span in spans2]) == {'LOC', 'DOUBLE_LOC'}