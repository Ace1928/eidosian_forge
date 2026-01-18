import copy
import datetime
import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
import yaml
from mistralclient.api.v2 import environments
from mistralclient.commands.v2 import environments as environment_cmd
from mistralclient.tests.unit import base
def test_update_from_json(self):
    env = copy.deepcopy(ENVIRONMENT_DICT)
    del env['created_at']
    del env['updated_at']
    self._test_update(jsonutils.dumps(env, indent=4))