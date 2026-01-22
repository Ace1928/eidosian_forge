import warnings
import weakref
import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.attrs import (
from spacy.lang.en import English
from spacy.lang.xx import MultiLanguage
from spacy.language import Language
from spacy.lexeme import Lexeme
from spacy.tokens import Doc, Span, SpanGroup, Token
from spacy.vocab import Vocab
from .test_underscore import clean_underscore  # noqa: F401
@Language.factory('my_pipe')
class CustomPipe:

    def __init__(self, nlp, name='my_pipe'):
        self.name = name
        Span.set_extension('my_ext', getter=self._get_my_ext)
        Doc.set_extension('my_ext', default=None)

    def __call__(self, doc):
        gathered_ext = []
        for sent in doc.sents:
            sent_ext = self._get_my_ext(sent)
            sent._.set('my_ext', sent_ext)
            gathered_ext.append(sent_ext)
        doc._.set('my_ext', '\n'.join(gathered_ext))
        return doc

    @staticmethod
    def _get_my_ext(span):
        return str(span.end)