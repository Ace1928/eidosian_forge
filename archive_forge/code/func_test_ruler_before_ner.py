import logging
import random
import pytest
from numpy.testing import assert_equal
from spacy import registry, util
from spacy.attrs import ENT_IOB
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.pipeline import EntityRecognizer
from spacy.pipeline._parser_internals.ner import BiluoPushDown
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example, iob_to_biluo, split_bilu_label
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_ruler_before_ner():
    """Test that an NER works after an entity_ruler: the second can add annotations"""
    nlp = English()
    patterns = [{'label': 'THING', 'pattern': 'This'}]
    ruler = nlp.add_pipe('entity_ruler')
    untrained_ner = nlp.add_pipe('ner')
    untrained_ner.add_label('MY_LABEL')
    nlp.initialize()
    ruler.add_patterns(patterns)
    doc = nlp('This is Antti Korhonen speaking in Finland')
    expected_iobs = ['B', 'O', 'O', 'O', 'O', 'O', 'O']
    expected_types = ['THING', '', '', '', '', '', '']
    assert [token.ent_iob_ for token in doc] == expected_iobs
    assert [token.ent_type_ for token in doc] == expected_types