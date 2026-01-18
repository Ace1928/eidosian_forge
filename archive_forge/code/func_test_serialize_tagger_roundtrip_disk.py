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
def test_serialize_tagger_roundtrip_disk(en_vocab, taggers):
    tagger1, tagger2 = taggers
    with make_tempdir() as d:
        file_path1 = d / 'tagger1'
        file_path2 = d / 'tagger2'
        tagger1.to_disk(file_path1)
        tagger2.to_disk(file_path2)
        cfg = {'model': DEFAULT_TAGGER_MODEL}
        model = registry.resolve(cfg, validate=True)['model']
        tagger1_d = Tagger(en_vocab, model).from_disk(file_path1)
        tagger2_d = Tagger(en_vocab, model).from_disk(file_path2)
        assert tagger1_d.to_bytes() == tagger2_d.to_bytes()