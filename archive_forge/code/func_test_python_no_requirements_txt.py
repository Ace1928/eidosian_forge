import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_no_requirements_txt(self):
    self.write_file('foo.py', '# python code')
    self.generate_configs(custom=True)
    self.assert_file_exists_with_contents('Dockerfile', self.DOCKERFILE_PREAMBLE + self.DOCKERFILE_VIRTUALENV_TEMPLATE.format(python_version='') + self.DOCKERFILE_INSTALL_APP + 'CMD my_entrypoint\n')
    self.assertEqual(set(os.listdir(self.temp_path)), {'foo.py', 'app.yaml', 'Dockerfile', '.dockerignore'})