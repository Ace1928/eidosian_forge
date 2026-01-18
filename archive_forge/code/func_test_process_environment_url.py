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
def test_process_environment_url(self, mock_url):
    env = b'\n        resource_registry:\n            "OS::Thingy": "a.yaml"\n        '
    url = 'http://no.where/some/path/to/file.yaml'
    tmpl_url = 'http://no.where/some/path/to/a.yaml'
    mock_url.side_effect = [io.BytesIO(env), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
    files, env_dict = template_utils.process_environment_and_files(url)
    self.assertEqual({'resource_registry': {'OS::Thingy': tmpl_url}}, env_dict)
    self.assertEqual(self.template_a.decode('utf-8'), files[tmpl_url])
    mock_url.assert_has_calls([mock.call(url), mock.call(tmpl_url), mock.call(tmpl_url)])