from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def assertRoundtripBlob(self, blob):
    raise NotImplementedError(self.assertRoundtripBlob)