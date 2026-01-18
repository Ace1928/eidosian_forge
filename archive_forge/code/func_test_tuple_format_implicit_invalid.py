import random
import numpy
import pytest
import srsly
from thinc.api import Adam, compounding
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.training import (
from spacy.training.align import get_alignments
from spacy.training.alignment_array import AlignmentArray
from spacy.training.converters import json_to_docs
from spacy.training.loop import train_while_improving
from spacy.util import (
from ..util import make_tempdir
def test_tuple_format_implicit_invalid():
    """Test that an error is thrown for an implicit invalid field"""
    train_data = [('Uber blew through $1 million a week', {'frumble': [(0, 4, 'ORG')]}), ('Spotify steps up Asia expansion', {'entities': [(0, 7, 'ORG'), (17, 21, 'LOC')]}), ('Google rebrands its business apps', {'entities': [(0, 6, 'ORG')]})]
    with pytest.raises(KeyError):
        _train_tuples(train_data)