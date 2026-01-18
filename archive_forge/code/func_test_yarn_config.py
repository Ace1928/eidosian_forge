import logging
import subprocess
from os.path import join as pjoin
from unittest.mock import patch
from jupyterlab import commands
from .test_jupyterlab import AppHandlerTest
def test_yarn_config(self):
    with patch('subprocess.check_output') as check_output:
        yarn_registry = 'https://private.yarn/manager'
        check_output.return_value = b'\n'.join([b'{"type":"info","data":"yarn config"}', b'{"type":"inspect","data":{"registry":"' + bytes(yarn_registry, 'utf-8') + b'"}}', b'{"type":"info","data":"npm config"}', b'{"type":"inspect","data":{"registry":"' + bytes(yarn_registry, 'utf-8') + b'"}}'])
        logger = logging.getLogger('jupyterlab')
        config = commands._yarn_config(logger)
        self.assertDictEqual(config, {'yarn config': {'registry': yarn_registry}, 'npm config': {'registry': yarn_registry}})