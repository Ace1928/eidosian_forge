import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
Test that Doc.from_array doesn't set heads that are out of bounds.