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
def test_process_multiple_environments_and_files(self, mock_url):
    env_file1 = '/home/my/dir/env1.yaml'
    env_file2 = '/home/my/dir/env2.yaml'
    env1 = b'\n        parameters:\n          "param1": "value1"\n        resource_registry:\n          "OS::Thingy1": "file:///home/b/a.yaml"\n        '
    env2 = b'\n        parameters:\n          "param2": "value2"\n        resource_registry:\n          "OS::Thingy2": "file:///home/b/b.yaml"\n        '
    mock_url.side_effect = [io.BytesIO(env1), io.BytesIO(self.template_a), io.BytesIO(self.template_a), io.BytesIO(env2), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
    files, env = template_utils.process_multiple_environments_and_files([env_file1, env_file2])
    self.assertEqual({'resource_registry': {'OS::Thingy1': 'file:///home/b/a.yaml', 'OS::Thingy2': 'file:///home/b/b.yaml'}, 'parameters': {'param1': 'value1', 'param2': 'value2'}}, env)
    self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/a.yaml'])
    self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/b.yaml'])
    mock_url.assert_has_calls([mock.call('file://%s' % env_file1), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/a.yaml'), mock.call('file://%s' % env_file2), mock.call('file:///home/b/b.yaml'), mock.call('file:///home/b/b.yaml')])