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
def test_roundtrip_docs_to_docbin(doc):
    text = doc.text
    idx = [t.idx for t in doc]
    tags = [t.tag_ for t in doc]
    pos = [t.pos_ for t in doc]
    morphs = [str(t.morph) for t in doc]
    lemmas = [t.lemma_ for t in doc]
    deps = [t.dep_ for t in doc]
    heads = [t.head.i for t in doc]
    cats = doc.cats
    ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    with make_tempdir() as tmpdir:
        reloaded_nlp = English()
        json_file = tmpdir / 'roundtrip.json'
        srsly.write_json(json_file, [docs_to_json(doc)])
        output_file = tmpdir / 'roundtrip.spacy'
        DocBin(docs=[doc]).to_disk(output_file)
        reader = Corpus(output_file)
        reloaded_examples = list(reader(reloaded_nlp))
    assert len(doc) == sum((len(eg) for eg in reloaded_examples))
    reloaded_example = reloaded_examples[0]
    assert text == reloaded_example.reference.text
    assert idx == [t.idx for t in reloaded_example.reference]
    assert tags == [t.tag_ for t in reloaded_example.reference]
    assert pos == [t.pos_ for t in reloaded_example.reference]
    assert morphs == [str(t.morph) for t in reloaded_example.reference]
    assert lemmas == [t.lemma_ for t in reloaded_example.reference]
    assert deps == [t.dep_ for t in reloaded_example.reference]
    assert heads == [t.head.i for t in reloaded_example.reference]
    assert ents == [(e.start_char, e.end_char, e.label_) for e in reloaded_example.reference.ents]
    assert 'TRAVEL' in reloaded_example.reference.cats
    assert 'BAKING' in reloaded_example.reference.cats
    assert cats['TRAVEL'] == reloaded_example.reference.cats['TRAVEL']
    assert cats['BAKING'] == reloaded_example.reference.cats['BAKING']