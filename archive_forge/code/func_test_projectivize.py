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
def test_projectivize(en_tokenizer):
    doc = en_tokenizer('He pretty quickly walks away')
    heads = [3, 2, 3, 3, 2]
    deps = ['dep'] * len(heads)
    example = Example.from_dict(doc, {'heads': heads, 'deps': deps})
    proj_heads, proj_labels = example.get_aligned_parse(projectivize=True)
    nonproj_heads, nonproj_labels = example.get_aligned_parse(projectivize=False)
    assert proj_heads == [3, 2, 3, 3, 3]
    assert nonproj_heads == [3, 2, 3, 3, 2]
    doc = en_tokenizer('Conrail')
    heads = [0]
    deps = ['dep']
    example = Example.from_dict(doc, {'heads': heads, 'deps': deps})
    proj_heads, proj_labels = example.get_aligned_parse(projectivize=True)
    assert proj_heads == heads
    assert proj_labels == deps
    doc_a = Doc(doc.vocab, words=['Double-Jointed'], spaces=[False], deps=['ROOT'], heads=[0])
    doc_b = Doc(doc.vocab, words=['Double', '-', 'Jointed'], spaces=[True, True, True], deps=['amod', 'punct', 'ROOT'], heads=[2, 2, 2])
    example = Example(doc_a, doc_b)
    proj_heads, proj_deps = example.get_aligned_parse(projectivize=True)
    assert proj_heads == [None]
    assert proj_deps == [None]