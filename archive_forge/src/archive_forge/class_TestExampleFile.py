import os.path
import subprocess
import sys
from unittest import mock
from oslo_config import cfg
from oslotest import base
from oslo_upgradecheck import upgradecheck
class TestExampleFile(base.BaseTestCase):

    def test_example_main(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../doc/source/main.py')
        self.assertEqual(upgradecheck.Code.FAILURE, subprocess.call([sys.executable, path, 'upgrade', 'check']))