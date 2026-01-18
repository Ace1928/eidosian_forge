import random
import numpy
import pytest
import srsly
from thinc.api import Adam, compounding
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.training import (
from spacy.training.align import get_alignments
from spacy.training.alignment_array import AlignmentArray
from spacy.training.converters import json_to_docs
from spacy.training.loop import train_while_improving
from spacy.util import (
from ..util import make_tempdir
def test_split_sentences(en_vocab):
    words = ['I', 'flew', 'to', 'San Francisco Valley', 'had', 'loads of fun']
    gold_words = ['I', 'flew', 'to', 'San', 'Francisco', 'Valley', 'had', 'loads', 'of', 'fun']
    sent_starts = [True, False, False, False, False, False, True, False, False, False]
    doc = Doc(en_vocab, words=words)
    example = Example.from_dict(doc, {'words': gold_words, 'sent_starts': sent_starts})
    assert example.text == 'I flew to San Francisco Valley had loads of fun '
    split_examples = example.split_sents()
    assert len(split_examples) == 2
    assert split_examples[0].text == 'I flew to San Francisco Valley '
    assert split_examples[1].text == 'had loads of fun '
    words = ['I', 'flew', 'to', 'San', 'Francisco', 'Valley', 'had', 'loads', 'of fun']
    gold_words = ['I', 'flew', 'to', 'San Francisco', 'Valley', 'had', 'loads of', 'fun']
    sent_starts = [True, False, False, False, False, True, False, False]
    doc = Doc(en_vocab, words=words)
    example = Example.from_dict(doc, {'words': gold_words, 'sent_starts': sent_starts})
    assert example.text == 'I flew to San Francisco Valley had loads of fun '
    split_examples = example.split_sents()
    assert len(split_examples) == 2
    assert split_examples[0].text == 'I flew to San Francisco Valley '
    assert split_examples[1].text == 'had loads of fun '