import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import approx
from spacy.lang.en import English
from spacy.scorer import PRFScore, ROCAUCScore, Scorer, _roc_auc_score, _roc_curve
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
def test_score_cats(en_tokenizer):
    text = 'some text'
    gold_doc = en_tokenizer(text)
    gold_doc.cats = {'POSITIVE': 1.0, 'NEGATIVE': 0.0}
    pred_doc = en_tokenizer(text)
    pred_doc.cats = {'POSITIVE': 0.75, 'NEGATIVE': 0.25}
    example = Example(pred_doc, gold_doc)
    scores1 = Scorer.score_cats([example], 'cats', labels=list(gold_doc.cats.keys()), multi_label=False, positive_label='POSITIVE', threshold=0.1)
    scores2 = Scorer.score_cats([example], 'cats', labels=list(gold_doc.cats.keys()), multi_label=False, positive_label='POSITIVE', threshold=0.9)
    assert scores1['cats_score'] == 1.0
    assert scores2['cats_score'] == 1.0
    assert scores1 == scores2
    scores = Scorer.score_cats([example], 'cats', labels=list(gold_doc.cats.keys()), multi_label=True, threshold=0.9)
    assert scores['cats_macro_f'] == 0.0
    scores = Scorer.score_cats([example], 'cats', labels=list(gold_doc.cats.keys()), multi_label=True, threshold=0.1)
    assert scores['cats_macro_f'] == 0.5