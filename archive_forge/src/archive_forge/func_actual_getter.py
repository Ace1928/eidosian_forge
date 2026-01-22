import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def actual_getter():
    return {'additionalProperties': False, 'required': ['name'], 'name': 'test_schema', 'properties': {'test': prop, 'readonly-test': prop_readonly}}