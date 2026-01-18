import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_no_requirements_txt_no_write(self):
    """Tests generate_config_data with no requirements.txt file."""
    self.write_file('foo.py', '# python code')
    cfg_files = self.generate_config_data(custom=True)
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', self.DOCKERFILE_PREAMBLE + self.DOCKERFILE_VIRTUALENV_TEMPLATE.format(python_version='') + self.DOCKERFILE_INSTALL_APP + 'CMD my_entrypoint\n')
    self.assertEqual(set(os.listdir(self.temp_path)), {'foo.py', 'app.yaml'})
    self.assertEqual({f.filename for f in cfg_files}, {'Dockerfile', '.dockerignore'})