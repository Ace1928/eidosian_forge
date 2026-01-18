from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def test_stack_checks_serializers_compatibility(self):
    repo = self.make_repository('repo', format=self.get_format())
    if getattr(repo._format, 'supports_tree_reference', False):
        matching_format_name = '2a'
        mismatching_format_name = 'rich-root-pack'
    elif repo.supports_rich_root():
        if repo._format.supports_chks:
            matching_format_name = '2a'
        else:
            matching_format_name = 'rich-root-pack'
        mismatching_format_name = 'pack-0.92-subtree'
    else:
        raise TestNotApplicable('No formats use non-v5 serializer without having rich-root also set')
    base = self.make_repository('base', format=matching_format_name)
    repo.add_fallback_repository(base)
    bad_repo = self.make_repository('mismatch', format=mismatching_format_name)
    e = self.assertRaises(errors.IncompatibleRepositories, repo.add_fallback_repository, bad_repo)
    self.assertContainsRe(str(e), '(?m)KnitPackRepository.*/mismatch/.*\\nis not compatible with\\n.*Repository.*/repo/.*\\ndifferent serializers')