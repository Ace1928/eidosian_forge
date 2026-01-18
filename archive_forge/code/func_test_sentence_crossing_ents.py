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
@pytest.mark.issue(7065)
@pytest.mark.parametrize('entity_in_first_sentence', [True, False])
def test_sentence_crossing_ents(entity_in_first_sentence: bool):
    """Tests if NEL crashes if entities cross sentence boundaries and the first associated sentence doesn't have an
    entity.
    entity_in_prior_sentence (bool): Whether to include an entity in the first sentence associated with the
    sentence-crossing entity.
    """
    nlp = English()
    vector_length = 3
    text = "Mahler 's Symphony No. 8 was beautiful."
    entities = [(10, 24, 'WORK')]
    links = {(10, 24): {'Q7304': 0.0, 'Q270853': 1.0}}
    if entity_in_first_sentence:
        entities.append((0, 6, 'PERSON'))
        links[0, 6] = {'Q7304': 1.0, 'Q270853': 0.0}
    sent_starts = [1, -1, 0, 0, 0, 1, 0, 0, 0]
    doc = nlp(text)
    example = Example.from_dict(doc, {'entities': entities, 'links': links, 'sent_starts': sent_starts})
    train_examples = [example]

    def create_kb(vocab):
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q270853', freq=12, entity_vector=[9, 1, -7])
        mykb.add_alias(alias='No. 8', entities=['Q270853'], probabilities=[1.0])
        mykb.add_entity(entity='Q7304', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias(alias='Mahler', entities=['Q7304'], probabilities=[1.0])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', last=True)
    entity_linker.set_kb(create_kb)
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(2):
        nlp.update(train_examples, sgd=optimizer)
    entity_linker.predict([example.reference])