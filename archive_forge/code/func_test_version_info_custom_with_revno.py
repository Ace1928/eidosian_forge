import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_version_info_custom_with_revno(self):
    output, error = self.run_bzr(['version-info', '--custom', '--template=VERSION_INFO r{revno})\n', 'gitr'], retcode=3)
    self.assertEqual(error, 'brz: ERROR: Variable {revno} is not available.\n')
    self.assertEqual(output, 'VERSION_INFO r')