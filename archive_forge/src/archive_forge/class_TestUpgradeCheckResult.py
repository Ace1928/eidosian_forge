import os.path
import subprocess
import sys
from unittest import mock
from oslo_config import cfg
from oslotest import base
from oslo_upgradecheck import upgradecheck
class TestUpgradeCheckResult(base.BaseTestCase):

    def test_details(self):
        result = upgradecheck.Result(upgradecheck.Code.SUCCESS, 'test details')
        self.assertEqual(0, result.code)
        self.assertEqual('test details', result.details)