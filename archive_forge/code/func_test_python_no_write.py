import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_no_write(self):
    """Tests generate_config_data with only requirements.txt.

        app.yaml should be written to disk, Dockerfile and .dockerignore
        returned by the method in memory. Tests that Dockerfile contents
        are correct.
        """
    self.write_file('requirements.txt', 'requirements')
    cfg_files = self.generate_config_data(deploy=True)
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', self.DOCKERFILE_PREAMBLE + self.DOCKERFILE_VIRTUALENV_TEMPLATE.format(python_version='') + self.DOCKERFILE_REQUIREMENTS_TXT + self.DOCKERFILE_INSTALL_APP + 'CMD my_entrypoint\n')
    self.assertEqual(set(os.listdir(self.temp_path)), {'requirements.txt', 'app.yaml'})
    self.assertEqual({f.filename for f in cfg_files}, {'Dockerfile', '.dockerignore'})