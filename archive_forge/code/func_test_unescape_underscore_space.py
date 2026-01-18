from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_unescape_underscore_space(self):
    self.assertEqual(b'bla _', unescape_file_id(b'bla_s__'))