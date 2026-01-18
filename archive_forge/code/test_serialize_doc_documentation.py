import copy
import pickle
import numpy
import pytest
from spacy.attrs import DEP, HEAD
from spacy.lang.en import English
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
Ensure that a modified pos attribute is serialized correctly.