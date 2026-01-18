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
def test_serialize_with_custom_tokenizer():
    """Test that serialization with custom tokenizer works without token_match.
    See: https://support.prodi.gy/t/how-to-save-a-custom-tokenizer/661/2
    """
    prefix_re = re.compile('1/|2/|:[0-9][0-9][A-K]:|:[0-9][0-9]:')
    suffix_re = re.compile('')
    infix_re = re.compile('[~]')

    def custom_tokenizer(nlp):
        return Tokenizer(nlp.vocab, {}, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer)
    nlp = Language()
    nlp.tokenizer = custom_tokenizer(nlp)
    with make_tempdir() as d:
        nlp.to_disk(d)