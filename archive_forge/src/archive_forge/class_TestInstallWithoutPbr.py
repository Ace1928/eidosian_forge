import os.path
import pkg_resources
import shlex
import sys
import fixtures
import testtools
import textwrap
from pbr.tests import base
from pbr.tests import test_packaging
class TestInstallWithoutPbr(base.BaseTestCase):

    @testtools.skipUnless(os.environ.get('PBR_INTEGRATION', None) == '1', 'integration tests not enabled')
    def test_install_without_pbr(self):
        tempdir = self.useFixture(fixtures.TempDir()).path
        dist_dir = os.path.join(tempdir, 'distdir')
        os.mkdir(dist_dir)
        self._run_cmd(sys.executable, ('setup.py', 'sdist', '-d', dist_dir), allow_fail=False, cwd=PBR_ROOT)
        test_pkg_dir = os.path.join(tempdir, 'testpkg')
        os.mkdir(test_pkg_dir)
        pkgs = {'pkgTest': {'setup.py': textwrap.dedent("                    #!/usr/bin/env python\n                    import setuptools\n                    setuptools.setup(\n                        name = 'pkgTest',\n                        tests_require = ['pkgReq'],\n                        test_suite='pkgReq'\n                    )\n                "), 'setup.cfg': textwrap.dedent('                    [easy_install]\n                    find_links = %s\n                ' % dist_dir)}, 'pkgReq': {'requirements.txt': textwrap.dedent('                    pbr\n                '), 'pkgReq/__init__.py': textwrap.dedent('                    print("FakeTest loaded and ran")\n                ')}}
        pkg_dirs = self.useFixture(test_packaging.CreatePackages(pkgs)).package_dirs
        test_pkg_dir = pkg_dirs['pkgTest']
        req_pkg_dir = pkg_dirs['pkgReq']
        self._run_cmd(sys.executable, ('setup.py', 'sdist', '-d', dist_dir), allow_fail=False, cwd=req_pkg_dir)
        venv = self.useFixture(test_packaging.Venv('nopbr', ['pip', 'wheel']))
        python = venv.python
        self.useFixture(base.CapturedSubprocess('nopbr', [python] + ['setup.py', 'test'], cwd=test_pkg_dir))