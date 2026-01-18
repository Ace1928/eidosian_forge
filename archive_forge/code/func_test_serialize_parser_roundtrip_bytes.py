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
@pytest.mark.parametrize('Parser', test_parsers)
def test_serialize_parser_roundtrip_bytes(en_vocab, Parser):
    cfg = {'model': DEFAULT_PARSER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    parser = Parser(en_vocab, model)
    new_parser = Parser(en_vocab, model)
    new_parser = new_parser.from_bytes(parser.to_bytes(exclude=['vocab']))
    bytes_2 = new_parser.to_bytes(exclude=['vocab'])
    bytes_3 = parser.to_bytes(exclude=['vocab'])
    assert len(bytes_2) == len(bytes_3)
    assert bytes_2 == bytes_3