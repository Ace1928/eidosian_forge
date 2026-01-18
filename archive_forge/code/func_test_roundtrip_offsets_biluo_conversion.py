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
def test_roundtrip_offsets_biluo_conversion(en_tokenizer):
    text = 'I flew to Silicon Valley via London.'
    biluo_tags = ['O', 'O', 'O', 'B-LOC', 'L-LOC', 'O', 'U-GPE', 'O']
    offsets = [(10, 24, 'LOC'), (29, 35, 'GPE')]
    doc = en_tokenizer(text)
    biluo_tags_converted = offsets_to_biluo_tags(doc, offsets)
    assert biluo_tags_converted == biluo_tags
    offsets_converted = biluo_tags_to_offsets(doc, biluo_tags)
    offsets_converted = [ent for ent in offsets if ent[2]]
    assert offsets_converted == offsets