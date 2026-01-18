import pytest
from numpy.testing import assert_almost_equal, assert_equal
from thinc.api import compounding, get_current_ops
from spacy import util
from spacy.attrs import TAG
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example
from ..util import make_tempdir
Test that Tagger.initialize() without gold tuples does not clobber
    the tag map.