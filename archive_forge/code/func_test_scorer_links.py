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
def test_scorer_links():
    train_examples = []
    nlp = English()
    ref1 = nlp('Julia lives in London happily.')
    ref1.ents = [Span(ref1, 0, 1, label='PERSON', kb_id='Q2'), Span(ref1, 3, 4, label='LOC', kb_id='Q3')]
    pred1 = nlp('Julia lives in London happily.')
    pred1.ents = [Span(pred1, 0, 1, label='PERSON', kb_id='Q70'), Span(pred1, 3, 4, label='LOC', kb_id='Q3')]
    train_examples.append(Example(pred1, ref1))
    ref2 = nlp('She loves London.')
    ref2.ents = [Span(ref2, 0, 1, label='PERSON', kb_id='Q2'), Span(ref2, 2, 3, label='LOC', kb_id='Q13')]
    pred2 = nlp('She loves London.')
    pred2.ents = [Span(pred2, 0, 1, label='PERSON', kb_id='Q2'), Span(pred2, 2, 3, label='LOC', kb_id='NIL')]
    train_examples.append(Example(pred2, ref2))
    ref3 = nlp('London is great.')
    ref3.ents = [Span(ref3, 0, 1, label='LOC', kb_id='NIL')]
    pred3 = nlp('London is great.')
    pred3.ents = [Span(pred3, 0, 1, label='LOC', kb_id='NIL')]
    train_examples.append(Example(pred3, ref3))
    scores = Scorer().score_links(train_examples, negative_labels=['NIL'])
    assert scores['nel_f_per_type']['PERSON']['p'] == 1 / 2
    assert scores['nel_f_per_type']['PERSON']['r'] == 1 / 2
    assert scores['nel_f_per_type']['LOC']['p'] == 1 / 1
    assert scores['nel_f_per_type']['LOC']['r'] == 1 / 2
    assert scores['nel_micro_p'] == 2 / 3
    assert scores['nel_micro_r'] == 2 / 4