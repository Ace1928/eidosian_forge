import pytest
from spacy.lang.en import English
from spacy_legacy.layers.staticvectors_v1 import StaticVectors_v1
@pytest.mark.parametrize('model_func,kwargs', [(StaticVectors_v1, {'nO': 128, 'nM': 300})])
def test_static_vector_v1(model_func, kwargs):
    nlp = English()
    model = model_func(**kwargs).initialize()
    for n_docs in range(3):
        docs = [nlp('') for _ in range(n_docs)]
        model.predict(docs)
        output, backprop = model.begin_update(docs)
        backprop(output)