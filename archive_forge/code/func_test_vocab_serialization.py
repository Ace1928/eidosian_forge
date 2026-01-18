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
def test_vocab_serialization(nlp):
    """Test that string information is retained across storage"""
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    mykb.add_entity(entity='Q1', freq=27, entity_vector=[1])
    q2_hash = mykb.add_entity(entity='Q2', freq=12, entity_vector=[2])
    mykb.add_entity(entity='Q3', freq=5, entity_vector=[3])
    mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.4, 0.1])
    adam_hash = mykb.add_alias(alias='adam', entities=['Q2'], probabilities=[0.9])
    candidates = mykb.get_alias_candidates('adam')
    assert len(candidates) == 1
    assert candidates[0].entity == q2_hash
    assert candidates[0].entity_ == 'Q2'
    assert candidates[0].alias == adam_hash
    assert candidates[0].alias_ == 'adam'
    with make_tempdir() as d:
        mykb.to_disk(d / 'kb')
        kb_new_vocab = InMemoryLookupKB(Vocab(), entity_vector_length=1)
        kb_new_vocab.from_disk(d / 'kb')
        candidates = kb_new_vocab.get_alias_candidates('adam')
        assert len(candidates) == 1
        assert candidates[0].entity == q2_hash
        assert candidates[0].entity_ == 'Q2'
        assert candidates[0].alias == adam_hash
        assert candidates[0].alias_ == 'adam'
        assert kb_new_vocab.get_vector('Q2') == [2]
        assert_almost_equal(kb_new_vocab.get_prior_prob('Q2', 'douglas'), 0.4)