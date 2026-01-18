import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
def test_overridden_rules_uncommented_in_yaml_file(self):
    converted_policy_data = self._test_convert_json_to_yaml_file()
    uncommented_overridden_rule = '# rule2_name\n"rule2_name": "rule:overridden"\n\n'
    self.assertIn(uncommented_overridden_rule, converted_policy_data)