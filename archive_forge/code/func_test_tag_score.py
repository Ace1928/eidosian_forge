import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import approx
from spacy.lang.en import English
from spacy.scorer import PRFScore, ROCAUCScore, Scorer, _roc_auc_score, _roc_curve
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
def test_tag_score(tagged_doc):
    scorer = Scorer()
    gold = {'tags': [t.tag_ for t in tagged_doc], 'pos': [t.pos_ for t in tagged_doc], 'morphs': [str(t.morph) for t in tagged_doc], 'sent_starts': [1 if t.is_sent_start else -1 for t in tagged_doc]}
    example = Example.from_dict(tagged_doc, gold)
    results = scorer.score([example])
    assert results['tag_acc'] == 1.0
    assert results['pos_acc'] == 1.0
    assert results['morph_acc'] == 1.0
    assert results['morph_micro_f'] == 1.0
    assert results['morph_per_feat']['NounType']['f'] == 1.0
    scorer = Scorer()
    tags = [t.tag_ for t in tagged_doc]
    tags[0] = 'NN'
    pos = [t.pos_ for t in tagged_doc]
    pos[1] = 'X'
    morphs = [str(t.morph) for t in tagged_doc]
    morphs[1] = 'Number=sing'
    morphs[2] = 'Number=plur'
    gold = {'tags': tags, 'pos': pos, 'morphs': morphs, 'sent_starts': gold['sent_starts']}
    example = Example.from_dict(tagged_doc, gold)
    results = scorer.score([example])
    assert results['tag_acc'] == 0.9
    assert results['pos_acc'] == 0.9
    assert results['morph_acc'] == approx(0.8)
    assert results['morph_micro_f'] == approx(0.8461538)
    assert results['morph_per_feat']['NounType']['f'] == 1.0
    assert results['morph_per_feat']['Poss']['f'] == 0.0
    assert results['morph_per_feat']['Number']['f'] == approx(0.72727272)
    scorer = Scorer()
    results = scorer.score([example], per_component=True)
    assert results['tagger']['tag_acc'] == 0.9
    assert results['morphologizer']['pos_acc'] == 0.9
    assert results['morphologizer']['morph_acc'] == approx(0.8)