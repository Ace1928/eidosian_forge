import pickle
import pytest
import srsly
from thinc.api import Linear
import spacy
from spacy import Vocab, load, registry
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import (
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL
from spacy.pipeline.tagger import DEFAULT_TAGGER_MODEL
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.tokens import Span
from spacy.util import ensure_path, load_model
from ..util import make_tempdir
def test_serialize_tagger_strings(en_vocab, de_vocab, taggers):
    label = 'SomeWeirdLabel'
    assert label not in en_vocab.strings
    assert label not in de_vocab.strings
    tagger = taggers[0]
    assert label not in tagger.vocab.strings
    with make_tempdir() as d:
        tagger.add_label(label)
        assert label in tagger.vocab.strings
        file_path = d / 'tagger1'
        tagger.to_disk(file_path)
        cfg = {'model': DEFAULT_TAGGER_MODEL}
        model = registry.resolve(cfg, validate=True)['model']
        tagger2 = Tagger(de_vocab, model).from_disk(file_path)
        assert label in tagger2.vocab.strings