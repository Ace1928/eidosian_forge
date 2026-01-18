from unittest import mock
import yaml
from heat.common import environment_format
from heat.tests import common
def test_yaml_none(self):
    self.assertEqual({}, environment_format.parse(None))