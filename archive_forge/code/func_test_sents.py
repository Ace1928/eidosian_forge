import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import approx
from spacy.lang.en import English
from spacy.scorer import PRFScore, ROCAUCScore, Scorer, _roc_auc_score, _roc_curve
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
def test_sents(sented_doc):
    scorer = Scorer()
    gold = {'sent_starts': [t.sent_start for t in sented_doc]}
    example = Example.from_dict(sented_doc, gold)
    scores = scorer.score([example])
    assert scores['sents_f'] == 1.0
    gold['sent_starts'][3] = 0
    gold['sent_starts'][4] = 1
    example = Example.from_dict(sented_doc, gold)
    scores = scorer.score([example])
    assert scores['sents_f'] == approx(0.3333333)