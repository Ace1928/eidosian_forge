import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.mark.parametrize('model_fixture', ['mle_bigram_model', 'mle_trigram_model', 'lidstone_bigram_model', 'laplace_bigram_model', 'wittenbell_trigram_model', 'absolute_discounting_trigram_model', 'kneserney_bigram_model', pytest.param('stupid_backoff_trigram_model', marks=pytest.mark.xfail(reason='Stupid Backoff is not a valid distribution'))])
@pytest.mark.parametrize('context', [('a',), ('c',), ('<s>',), ('b',), ('<UNK>',), ('d',), ('e',), ('r',), ('w',)], ids=itemgetter(0))
def test_sums_to_1(model_fixture, context, request):
    model = request.getfixturevalue(model_fixture)
    scores_for_context = sum((model.score(w, context) for w in model.vocab))
    assert pytest.approx(scores_for_context, 1e-07) == 1.0