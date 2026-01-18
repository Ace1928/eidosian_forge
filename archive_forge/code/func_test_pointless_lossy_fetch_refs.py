from ...controldir import format_registry
from ...repository import InterRepository
from ...tests import TestCaseWithTransport
from ..interrepo import InterToGitRepository
from ..mapping import BzrGitMappingExperimental, BzrGitMappingv1
def test_pointless_lossy_fetch_refs(self):
    revidmap, old_refs, new_refs = self._get_interrepo().fetch_refs(lambda x: {}, lossy=True)
    self.assertEqual(old_refs, {b'HEAD': (b'ref: refs/heads/master', None)})
    self.assertEqual(new_refs, {})
    self.assertEqual(revidmap, {})