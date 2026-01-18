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
def test_serialize_sentencerecognizer(en_vocab):
    cfg = {'model': DEFAULT_SENTER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    sr = SentenceRecognizer(en_vocab, model)
    sr_b = sr.to_bytes()
    sr_d = SentenceRecognizer(en_vocab, model).from_bytes(sr_b)
    assert sr.to_bytes() == sr_d.to_bytes()