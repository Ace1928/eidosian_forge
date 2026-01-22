from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
class FileidTests(tests.TestCase):

    def test_escape_space(self):
        self.assertEqual(b'bla_s', escape_file_id(b'bla '))

    def test_escape_control_l(self):
        self.assertEqual(b'bla_c', escape_file_id(b'bla\x0c'))

    def test_unescape_control_l(self):
        self.assertEqual(b'bla\x0c', unescape_file_id(b'bla_c'))

    def test_escape_underscore(self):
        self.assertEqual(b'bla__', escape_file_id(b'bla_'))

    def test_escape_underscore_space(self):
        self.assertEqual(b'bla___s', escape_file_id(b'bla_ '))

    def test_unescape_underscore(self):
        self.assertEqual(b'bla ', unescape_file_id(b'bla_s'))

    def test_unescape_underscore_space(self):
        self.assertEqual(b'bla _', unescape_file_id(b'bla_s__'))