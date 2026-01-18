import pytest
from thinc.api import Adam
from spacy import registry
from spacy.attrs import NORM
from spacy.pipeline import DependencyParser
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_sents_1_3(parser):
    doc = Doc(parser.vocab, words=['a', 'b', 'c', 'd'])
    doc[0].is_sent_start = True
    doc[1].is_sent_start = True
    doc[2].is_sent_start = None
    doc[3].is_sent_start = True
    doc = parser(doc)
    assert len(list(doc.sents)) >= 3
    doc = Doc(parser.vocab, words=['a', 'b', 'c', 'd'])
    doc[0].is_sent_start = True
    doc[1].is_sent_start = True
    doc[2].is_sent_start = False
    doc[3].is_sent_start = True
    doc = parser(doc)
    assert len(list(doc.sents)) == 3