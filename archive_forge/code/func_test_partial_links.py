from typing import Any, Callable, Dict, Iterable, Tuple
import pytest
from numpy.testing import assert_equal
from spacy import Language, registry, util
from spacy.attrs import ENT_KB_ID
from spacy.compat import pickle
from spacy.kb import Candidate, InMemoryLookupKB, KnowledgeBase, get_candidates
from spacy.lang.en import English
from spacy.ml import load_kb
from spacy.ml.models.entity_linker import build_span_maker
from spacy.pipeline import EntityLinker
from spacy.pipeline.legacy import EntityLinker_v1
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.scorer import Scorer
from spacy.tests.util import make_tempdir
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import ensure_path
from spacy.vocab import Vocab
def test_partial_links():
    TRAIN_DATA = [('Russ Cochran his reprints include EC Comics.', {'links': {(0, 12): {'Q2146908': 1.0}}, 'entities': [(0, 12, 'PERSON')], 'sent_starts': [1, -1, 0, 0, 0, 0, 0, 0]})]
    nlp = English()
    vector_length = 3
    train_examples = []
    for text, annotation in TRAIN_DATA:
        doc = nlp(text)
        train_examples.append(Example.from_dict(doc, annotation))

    def create_kb(vocab):
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias('Russ Cochran', ['Q2146908'], [0.9])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', last=True)
    entity_linker.set_kb(create_kb)
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    nlp.add_pipe('sentencizer', first=True)
    patterns = [{'label': 'PERSON', 'pattern': [{'LOWER': 'russ'}, {'LOWER': 'cochran'}]}, {'label': 'ORG', 'pattern': [{'LOWER': 'ec'}, {'LOWER': 'comics'}]}]
    ruler = nlp.add_pipe('entity_ruler', before='entity_linker')
    ruler.add_patterns(patterns)
    results = nlp.evaluate(train_examples)
    assert 'PERSON' in results['ents_per_type']
    assert 'PERSON' in results['nel_f_per_type']
    assert 'ORG' in results['ents_per_type']
    assert 'ORG' not in results['nel_f_per_type']