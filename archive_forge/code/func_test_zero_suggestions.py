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
@pytest.mark.parametrize('name', SPANCAT_COMPONENTS)
def test_zero_suggestions(name):

    @registry.misc('test_mixed_zero_suggester')
    def make_mixed_zero_suggester():

        def mixed_zero_suggester(docs, *, ops=None):
            if ops is None:
                ops = get_current_ops()
            spans = []
            lengths = []
            for doc in docs:
                if len(doc) > 0 and len(doc) % 2 == 0:
                    spans.append((0, 1))
                    lengths.append(1)
                else:
                    lengths.append(0)
            spans = ops.asarray2i(spans)
            lengths_array = ops.asarray1i(lengths)
            if len(spans) > 0:
                output = Ragged(ops.xp.vstack(spans), lengths_array)
            else:
                output = Ragged(ops.xp.zeros((0, 0), dtype='i'), lengths_array)
            return output
        return mixed_zero_suggester
    fix_random_seed(0)
    nlp = English()
    spancat = nlp.add_pipe(name, config={'suggester': {'@misc': 'test_mixed_zero_suggester'}, 'spans_key': SPAN_KEY})
    train_examples = make_examples(nlp)
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    assert spancat.model.get_dim('nO') == spancat._n_labels
    assert set(spancat.labels) == {'LOC', 'PERSON'}
    nlp.update(train_examples, sgd=optimizer)
    nlp('')
    nlp('one')
    nlp('two two')
    list(nlp.pipe(['one', 'two two', 'three three three', '', 'four four four four']))
    list(nlp.pipe(['', 'one', 'three three three']))