import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_custom_runtime(self):
    self.write_file('test.py', 'test file')
    self.generate_configs(custom=True)
    with open(os.path.join(self.temp_path, 'app.yaml')) as f:
        app_yaml_contents = f.read()
    self.assertMultiLineEqual(app_yaml_contents, textwrap.dedent('                entrypoint: my_entrypoint\n                env: flex\n                runtime: custom\n                '))
    self.assertEqual(set(os.listdir(self.temp_path)), {'test.py', 'app.yaml', '.dockerignore', 'Dockerfile'})