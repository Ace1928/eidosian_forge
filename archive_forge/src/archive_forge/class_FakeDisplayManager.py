import collections
import io
import sys
from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient.tests.unit import utils as test_utils
from cinderclient import utils
class FakeDisplayManager(FakeManager):
    resource_class = FakeDisplayResource
    resources = [FakeDisplayResource('4242', {'display_name': 'entity_three'})]