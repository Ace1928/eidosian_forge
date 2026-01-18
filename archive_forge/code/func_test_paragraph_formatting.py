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
def test_paragraph_formatting(self):
    description = "\nHere's a neat description with a paragraph. We want to make sure that it wraps\nproperly.\n"
    expected = '# Here\'s a neat description with a paragraph. We want to make sure\n# that it wraps properly.\n#"admin": "is_admin:True"\n\n'
    self._test_formatting(description, expected)