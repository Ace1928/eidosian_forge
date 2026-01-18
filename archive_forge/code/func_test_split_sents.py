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
def test_split_sents(merged_dict):
    nlp = English()
    example = Example.from_dict(Doc(nlp.vocab, words=merged_dict['words'], spaces=merged_dict['spaces']), merged_dict)
    assert example.text == 'Hi there everyone It is just me'
    split_examples = example.split_sents()
    assert len(split_examples) == 2
    assert split_examples[0].text == 'Hi there everyone '
    assert split_examples[1].text == 'It is just me'
    token_annotation_1 = split_examples[0].to_dict()['token_annotation']
    assert token_annotation_1['ORTH'] == ['Hi', 'there', 'everyone']
    assert token_annotation_1['TAG'] == ['INTJ', 'ADV', 'PRON']
    assert token_annotation_1['SENT_START'] == [1, 0, 0]
    token_annotation_2 = split_examples[1].to_dict()['token_annotation']
    assert token_annotation_2['ORTH'] == ['It', 'is', 'just', 'me']
    assert token_annotation_2['TAG'] == ['PRON', 'AUX', 'ADV', 'PRON']
    assert token_annotation_2['SENT_START'] == [1, 0, 0, 0]