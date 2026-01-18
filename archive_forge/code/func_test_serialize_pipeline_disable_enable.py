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
def test_serialize_pipeline_disable_enable():
    nlp = English()
    nlp.add_pipe('ner')
    nlp.add_pipe('tagger')
    nlp.disable_pipe('tagger')
    assert nlp.config['nlp']['disabled'] == ['tagger']
    config = nlp.config.copy()
    nlp2 = English.from_config(config)
    assert nlp2.pipe_names == ['ner']
    assert nlp2.component_names == ['ner', 'tagger']
    assert nlp2.disabled == ['tagger']
    assert nlp2.config['nlp']['disabled'] == ['tagger']
    with make_tempdir() as d:
        nlp2.to_disk(d)
        nlp3 = spacy.load(d)
    assert nlp3.pipe_names == ['ner']
    assert nlp3.component_names == ['ner', 'tagger']
    with make_tempdir() as d:
        nlp3.to_disk(d)
        nlp4 = spacy.load(d, disable=['ner'])
    assert nlp4.pipe_names == []
    assert nlp4.component_names == ['ner', 'tagger']
    assert nlp4.disabled == ['ner', 'tagger']
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp5 = spacy.load(d, exclude=['tagger'])
    assert nlp5.pipe_names == ['ner']
    assert nlp5.component_names == ['ner']
    assert nlp5.disabled == []