import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
class ShallowTests(ExternalBase):

    def setUp(self):
        super().setUp()
        self.repo = GitRepo.init('gitr', mkdir=True)
        self.build_tree_contents([('gitr/foo', b'hello from git')])
        self.repo.stage('foo')
        self.repo.do_commit(b'message', committer=b'Somebody <user@example.com>', author=b'Somebody <user@example.com>', commit_timestamp=1526330165, commit_timezone=0, author_timestamp=1526330165, author_timezone=0, merge_heads=[b'aa' * 20])

    def test_log_shallow(self):
        output, error = self.run_bzr(['log', 'gitr'], retcode=3)
        self.assertEqual(error, 'brz: ERROR: Further revision history missing.\n')
        self.assertEqual(output, '------------------------------------------------------------\nrevision-id: git-v1:' + self.repo.head().decode('ascii') + '\ngit commit: ' + self.repo.head().decode('ascii') + '\ncommitter: Somebody <user@example.com>\ntimestamp: Mon 2018-05-14 20:36:05 +0000\nmessage:\n  message\n')

    def test_version_info_rio(self):
        output, error = self.run_bzr(['version-info', '--rio', 'gitr'])
        self.assertEqual(error, '')
        self.assertNotIn('revno:', output)

    def test_version_info_python(self):
        output, error = self.run_bzr(['version-info', '--python', 'gitr'])
        self.assertEqual(error, '')
        self.assertNotIn('revno:', output)

    def test_version_info_custom_with_revno(self):
        output, error = self.run_bzr(['version-info', '--custom', '--template=VERSION_INFO r{revno})\n', 'gitr'], retcode=3)
        self.assertEqual(error, 'brz: ERROR: Variable {revno} is not available.\n')
        self.assertEqual(output, 'VERSION_INFO r')

    def test_version_info_custom_without_revno(self):
        output, error = self.run_bzr(['version-info', '--custom', '--template=VERSION_INFO \n', 'gitr'])
        self.assertEqual(error, '')
        self.assertEqual(output, 'VERSION_INFO \n')