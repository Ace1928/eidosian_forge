import sys
import pytest
from numpy.testing import assert_equal, suppress_warnings
from scipy._lib import doccer
def test_unindent():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        assert_equal(doccer.unindent_string(param_doc1), param_doc1)
        assert_equal(doccer.unindent_string(param_doc2), param_doc2)
        assert_equal(doccer.unindent_string(param_doc3), param_doc1)