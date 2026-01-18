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
def test_serialize_language_meta_disk(meta_data):
    language = Language(meta=meta_data)
    with make_tempdir() as d:
        language.to_disk(d)
        new_language = Language().from_disk(d)
    assert new_language.meta == language.meta