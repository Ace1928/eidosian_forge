import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import approx
from spacy.lang.en import English
from spacy.scorer import PRFScore, ROCAUCScore, Scorer, _roc_auc_score, _roc_curve
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
def test_score_spans():
    nlp = English()
    text = 'This is just a random sentence.'
    key = 'my_spans'
    gold = nlp.make_doc(text)
    pred = nlp.make_doc(text)
    spans = []
    spans.append(gold.char_span(0, 4, label='PERSON'))
    spans.append(gold.char_span(0, 7, label='ORG'))
    spans.append(gold.char_span(8, 12, label='ORG'))
    gold.spans[key] = spans

    def span_getter(doc, span_key):
        return doc.spans[span_key]
    pred.spans[key] = gold.spans[key].copy(doc=pred)
    eg = Example(pred, gold)
    scores = Scorer.score_spans([eg], attr=key, getter=span_getter)
    assert scores[f'{key}_p'] == 1.0
    assert scores[f'{key}_r'] < 1.0
    pred.spans[key] = gold.spans[key].copy(doc=pred)
    eg = Example(pred, gold)
    scores = Scorer.score_spans([eg], attr=key, getter=span_getter, allow_overlap=True)
    assert scores[f'{key}_p'] == 1.0
    assert scores[f'{key}_r'] == 1.0
    new_spans = [Span(pred, span.start, span.end, label='WRONG') for span in spans]
    pred.spans[key] = new_spans
    eg = Example(pred, gold)
    scores = Scorer.score_spans([eg], attr=key, getter=span_getter, allow_overlap=True)
    assert scores[f'{key}_p'] == 0.0
    assert scores[f'{key}_r'] == 0.0
    assert f'{key}_per_type' in scores
    scores = Scorer.score_spans([eg], attr=key, getter=span_getter, allow_overlap=True, labeled=False)
    assert scores[f'{key}_p'] == 1.0
    assert scores[f'{key}_r'] == 1.0
    assert f'{key}_per_type' not in scores