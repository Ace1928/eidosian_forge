import pytest
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.tokens.underscore import Underscore
Test that custom extensions are correctly serialized in DocBin.