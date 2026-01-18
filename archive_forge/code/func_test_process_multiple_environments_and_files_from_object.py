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
def test_process_multiple_environments_and_files_from_object(self):
    env_object = 'http://no.where/path/to/env.yaml'
    env1 = b'\n        parameters:\n          "param1": "value1"\n        resource_registry:\n          "OS::Thingy1": "b/a.yaml"\n        '
    self.object_requested = False

    def env_path_is_object(object_url):
        return True

    def object_request(method, object_url):
        self.object_requested = True
        self.assertEqual('GET', method)
        self.assertTrue(object_url.startswith('http://no.where/path/to/'))
        if object_url == env_object:
            return env1
        else:
            return self.template_a
    files, env = template_utils.process_multiple_environments_and_files(env_paths=[env_object], env_path_is_object=env_path_is_object, object_request=object_request)
    self.assertEqual({'resource_registry': {'OS::Thingy1': 'http://no.where/path/to/b/a.yaml'}, 'parameters': {'param1': 'value1'}}, env)
    self.assertEqual(self.template_a.decode('utf-8'), files['http://no.where/path/to/b/a.yaml'])