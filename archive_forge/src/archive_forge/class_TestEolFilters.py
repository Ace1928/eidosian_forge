from .. import errors
from ..filters import _get_filter_stack_for
from ..filters.eol import _to_crlf_converter, _to_lf_converter
from . import TestCase
class TestEolFilters(TestCase):

    def test_to_lf(self):
        result = _to_lf_converter([_sample_file1])
        self.assertEqual([b'hello\nworld\n'], result)

    def test_to_crlf(self):
        result = _to_crlf_converter([_sample_file1])
        self.assertEqual([b'hello\r\nworld\r\n'], result)