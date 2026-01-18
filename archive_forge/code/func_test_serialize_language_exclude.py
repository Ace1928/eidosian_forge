import pickle
import re
import pytest
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.training import Example
from spacy.util import load_config_from_str
from ..util import make_tempdir
def test_serialize_language_exclude(meta_data):
    name = 'name-in-fixture'
    nlp = Language(meta=meta_data)
    assert nlp.meta['name'] == name
    new_nlp = Language().from_bytes(nlp.to_bytes())
    assert new_nlp.meta['name'] == name
    new_nlp = Language().from_bytes(nlp.to_bytes(), exclude=['meta'])
    assert not new_nlp.meta['name'] == name
    new_nlp = Language().from_bytes(nlp.to_bytes(exclude=['meta']))
    assert not new_nlp.meta['name'] == name