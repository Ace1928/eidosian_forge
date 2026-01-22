import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def assertMessageRoundtrips(self, message):
    """Assert that message roundtrips to a repository and back intact."""
    tree = self.make_branch_and_tree('.')
    a = tree.commit(message, allow_pointless=True)
    rev = tree.branch.repository.get_revision(a)
    serializer = getattr(tree.branch.repository, '_serializer', None)
    if serializer is not None and serializer.squashes_xml_invalid_characters:
        escaped_message, escape_count = re.subn('[^\t\n\r -\ud7ff\ue000-�]+', lambda match: match.group(0).encode('unicode_escape').decode('ascii'), message)
        self.assertEqual(rev.message, escaped_message)
    else:
        self.assertEqual(rev.message, message)
    self.assertIsInstance(rev.message, str)