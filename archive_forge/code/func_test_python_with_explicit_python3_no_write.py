import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_with_explicit_python3_no_write(self):
    """Tests generate_config_data with python version '3' in appinfo."""
    self.write_file('test.py', 'test file')
    config = testutil.AppInfoFake(runtime='python', entrypoint='run_me_some_python!', runtime_config=dict(python_version='3'))
    cfg_files = self.generate_config_data(appinfo=config, deploy=True)
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', self.DOCKERFILE_PREAMBLE + self.DOCKERFILE_VIRTUALENV_TEMPLATE.format(python_version='3.6') + self.DOCKERFILE_INSTALL_APP + 'CMD run_me_some_python!\n')
    self.assertEqual(set(os.listdir(self.temp_path)), {'test.py'})
    self.assertEqual({f.filename for f in cfg_files}, {'Dockerfile', '.dockerignore'})