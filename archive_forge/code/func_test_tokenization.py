import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import approx
from spacy.lang.en import English
from spacy.scorer import PRFScore, ROCAUCScore, Scorer, _roc_auc_score, _roc_curve
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
def test_tokenization(sented_doc):
    scorer = Scorer()
    gold = {'sent_starts': [t.sent_start for t in sented_doc]}
    example = Example.from_dict(sented_doc, gold)
    scores = scorer.score([example])
    assert scores['token_acc'] == 1.0
    nlp = English()
    example.predicted = Doc(nlp.vocab, words=['One', 'sentence.', 'Two', 'sentences.', 'Three', 'sentences.'], spaces=[True, True, True, True, True, False])
    example.predicted[1].is_sent_start = False
    scores = scorer.score([example])
    assert scores['token_acc'] == 0.5
    assert scores['token_p'] == 0.5
    assert scores['token_r'] == approx(0.33333333)
    assert scores['token_f'] == 0.4
    scorer = Scorer()
    scores = scorer.score([example], per_component=True)
    assert scores['tokenizer']['token_acc'] == 0.5
    assert scores['tokenizer']['token_p'] == 0.5
    assert scores['tokenizer']['token_r'] == approx(0.33333333)
    assert scores['tokenizer']['token_f'] == 0.4