import json
import tempfile
from unittest import mock
import io
from oslo_serialization import base64
import testtools
from testtools import matchers
from urllib import error
import yaml
from heatclient.common import template_utils
from heatclient.common import utils
from heatclient import exc
@mock.patch('urllib.request.urlopen')
def test_process_environment_empty_file(self, mock_url):
    env_file = '/home/my/dir/env.yaml'
    env = b''
    mock_url.return_value = io.BytesIO(env)
    files, env_dict = template_utils.process_environment_and_files(env_file)
    self.assertEqual({}, env_dict)
    self.assertEqual({}, files)
    mock_url.assert_called_with('file://%s' % env_file)