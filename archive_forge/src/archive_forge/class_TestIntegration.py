import os.path
import pkg_resources
import shlex
import sys
import fixtures
import testtools
import textwrap
from pbr.tests import base
from pbr.tests import test_packaging
class TestIntegration(base.BaseTestCase):
    scenarios = list(all_projects())

    def setUp(self):
        env = fixtures.EnvironmentVariable('OS_TEST_TIMEOUT', os.environ.get('OS_TEST_TIMEOUT', '600'))
        with env:
            super(TestIntegration, self).setUp()
        base._config_git()

    @testtools.skipUnless(os.environ.get('PBR_INTEGRATION', None) == '1', 'integration tests not enabled')
    def test_integration(self):
        path = os.path.join(REPODIR, self.short_name)
        setup_cfg = os.path.join(path, 'setup.cfg')
        project_name = pkg_resources.safe_name(self.short_name).lower()
        if os.path.exists(setup_cfg):
            config = configparser.ConfigParser()
            config.read(setup_cfg)
            if config.has_section('metadata'):
                raw_name = config.get('metadata', 'name', fallback='notapackagename')
                project_name = pkg_resources.safe_name(raw_name).lower()
        constraints = os.path.join(REPODIR, 'requirements', 'upper-constraints.txt')
        tmp_constraints = os.path.join(self.useFixture(fixtures.TempDir()).path, 'upper-constraints.txt')
        with open(constraints, 'r') as src:
            with open(tmp_constraints, 'w') as dest:
                for line in src:
                    constraint = line.split('===')[0]
                    if project_name != constraint:
                        dest.write(line)
        pip_cmd = PIP_CMD + ['-c', tmp_constraints]
        venv = self.useFixture(test_packaging.Venv('sdist', modules=['pip', 'wheel', PBRVERSION], pip_cmd=PIP_CMD))
        python = venv.python
        self.useFixture(base.CapturedSubprocess('sdist', [python, 'setup.py', 'sdist'], cwd=path))
        venv = self.useFixture(test_packaging.Venv('tarball', modules=['pip', 'wheel', PBRVERSION], pip_cmd=PIP_CMD))
        python = venv.python
        filename = os.path.join(path, 'dist', os.listdir(os.path.join(path, 'dist'))[0])
        self.useFixture(base.CapturedSubprocess('tarball', [python] + pip_cmd + [filename]))
        venv = self.useFixture(test_packaging.Venv('install-git', modules=['pip', 'wheel', PBRVERSION], pip_cmd=PIP_CMD))
        root = venv.path
        python = venv.python
        self.useFixture(base.CapturedSubprocess('install-git', [python] + pip_cmd + ['git+file://' + path]))
        if self.short_name == 'nova':
            found = False
            for _, _, filenames in os.walk(root):
                if 'alembic.ini' in filenames:
                    found = True
            self.assertTrue(found)
        venv = self.useFixture(test_packaging.Venv('install-e', modules=['pip', 'wheel', PBRVERSION], pip_cmd=PIP_CMD))
        root = venv.path
        python = venv.python
        self.useFixture(base.CapturedSubprocess('install-e', [python] + pip_cmd + ['-e', path]))