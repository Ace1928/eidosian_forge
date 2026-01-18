import os.path
import pkg_resources
import shlex
import sys
import fixtures
import testtools
import textwrap
from pbr.tests import base
from pbr.tests import test_packaging
@testtools.skipUnless(os.environ.get('PBR_INTEGRATION', None) == '1', 'integration tests not enabled')
def test_pip_versions(self):
    pkgs = {'test_markers': {'requirements.txt': textwrap.dedent("                    pkg_a; python_version=='1.2'\n                    pkg_b; python_version!='1.2'\n                ")}, 'pkg_a': {}, 'pkg_b': {}}
    pkg_dirs = self.useFixture(test_packaging.CreatePackages(pkgs)).package_dirs
    temp_dir = self.useFixture(fixtures.TempDir()).path
    repo_dir = os.path.join(temp_dir, 'repo')
    venv = self.useFixture(test_packaging.Venv('markers'))
    bin_python = venv.python
    os.mkdir(repo_dir)
    for module in self.modules:
        self._run_cmd(bin_python, ['-m', 'pip', 'install', '--upgrade', module], cwd=venv.path, allow_fail=False)
    for pkg in pkg_dirs:
        self._run_cmd(bin_python, ['setup.py', 'sdist', '-d', repo_dir], cwd=pkg_dirs[pkg], allow_fail=False)
    self._run_cmd(bin_python, ['-m', 'pip', 'install', '--no-index', '-f', repo_dir, 'test_markers'], cwd=venv.path, allow_fail=False)
    self.assertIn('pkg-b', self._run_cmd(bin_python, ['-m', 'pip', 'freeze'], cwd=venv.path, allow_fail=False)[0])