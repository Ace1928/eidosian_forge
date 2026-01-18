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
def test_serialize_tagger_roundtrip_bytes(en_vocab, taggers):
    tagger1 = taggers[0]
    tagger1_b = tagger1.to_bytes()
    tagger1 = tagger1.from_bytes(tagger1_b)
    assert tagger1.to_bytes() == tagger1_b
    cfg = {'model': DEFAULT_TAGGER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    new_tagger1 = Tagger(en_vocab, model).from_bytes(tagger1_b)
    new_tagger1_b = new_tagger1.to_bytes()
    assert len(new_tagger1_b) == len(tagger1_b)
    assert new_tagger1_b == tagger1_b