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
def test_process_environment_relative_file(self, mock_url):
    env_file = '/home/my/dir/env.yaml'
    env_url = 'file:///home/my/dir/env.yaml'
    env = b'\n        resource_registry:\n          "OS::Thingy": a.yaml\n        '
    mock_url.side_effect = [io.BytesIO(env), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
    self.assertEqual(env_url, utils.normalise_file_path_to_url(env_file))
    self.assertEqual('file:///home/my/dir', utils.base_url_for_url(env_url))
    files, env_dict = template_utils.process_environment_and_files(env_file)
    self.assertEqual({'resource_registry': {'OS::Thingy': 'file:///home/my/dir/a.yaml'}}, env_dict)
    self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/my/dir/a.yaml'])
    mock_url.assert_has_calls([mock.call(env_url), mock.call('file:///home/my/dir/a.yaml'), mock.call('file:///home/my/dir/a.yaml')])