import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_to_json_underscore_doc_getters(doc):

    def get_text_length(doc):
        return len(doc.text)
    Doc.set_extension('text_length', getter=get_text_length)
    doc_json = doc.to_json(underscore=['text_length'])
    assert doc_json['_']['text_length'] == get_text_length(doc)