import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_app_yaml_no_entrypoint_no_write(self):
    """Tests generate_config_data with fake appinfo, no entrypoint."""
    self.write_file('test.py', 'test file')
    config = testutil.AppInfoFake(runtime='python')
    cfg_files = self.generate_config_data(appinfo=config, deploy=True)
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', self.DOCKERFILE_PREAMBLE + self.DOCKERFILE_VIRTUALENV_TEMPLATE.format(python_version='') + self.DOCKERFILE_INSTALL_APP + 'CMD my_entrypoint\n')
    self.assertEqual(set(os.listdir(self.temp_path)), {'test.py'})
    self.assertEqual({f.filename for f in cfg_files}, {'Dockerfile', '.dockerignore'})