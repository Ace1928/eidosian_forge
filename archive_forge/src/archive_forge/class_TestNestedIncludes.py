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
class TestNestedIncludes(testtools.TestCase):
    hot_template = b'heat_template_version: 2013-05-23\nparameters:\n  param1:\n    type: string\nresources:\n  resource1:\n    type: foo.yaml\n    properties:\n      foo: bar\n  resource2:\n    type: OS::Heat::ResourceGroup\n    properties:\n      resource_def:\n        type: spam/egg.yaml\n      with: {get_file: spam/ham.yaml}\n    '
    egg_template = b'heat_template_version: 2013-05-23\nparameters:\n  param1:\n    type: string\nresources:\n  resource1:\n    type: one.yaml\n    properties:\n      foo: bar\n  resource2:\n    type: OS::Heat::ResourceGroup\n    properties:\n      resource_def:\n        type: two.yaml\n      with: {get_file: three.yaml}\n    '
    foo_template = b'heat_template_version: "2013-05-23"\nparameters:\n  foo:\n    type: string\n    '

    @mock.patch('urllib.request.urlopen')
    def test_env_nested_includes(self, mock_url):
        env_file = '/home/my/dir/env.yaml'
        env_url = 'file:///home/my/dir/env.yaml'
        env = b'\n        resource_registry:\n          "OS::Thingy": template.yaml\n        '
        template_url = 'file:///home/my/dir/template.yaml'
        foo_url = 'file:///home/my/dir/foo.yaml'
        egg_url = 'file:///home/my/dir/spam/egg.yaml'
        ham_url = 'file:///home/my/dir/spam/ham.yaml'
        one_url = 'file:///home/my/dir/spam/one.yaml'
        two_url = 'file:///home/my/dir/spam/two.yaml'
        three_url = 'file:///home/my/dir/spam/three.yaml'

        def side_effect(args):
            if env_url == args:
                return io.BytesIO(env)
            if template_url == args:
                return io.BytesIO(self.hot_template)
            if foo_url == args:
                return io.BytesIO(self.foo_template)
            if egg_url == args:
                return io.BytesIO(self.egg_template)
            if ham_url == args:
                return io.BytesIO(b'ham contents')
            if one_url == args:
                return io.BytesIO(self.foo_template)
            if two_url == args:
                return io.BytesIO(self.foo_template)
            if three_url == args:
                return io.BytesIO(b'three contents')
        mock_url.side_effect = side_effect
        files, env_dict = template_utils.process_environment_and_files(env_file)
        self.assertEqual({'resource_registry': {'OS::Thingy': template_url}}, env_dict)
        self.assertEqual({'heat_template_version': '2013-05-23', 'parameters': {'param1': {'type': 'string'}}, 'resources': {'resource1': {'properties': {'foo': 'bar'}, 'type': foo_url}, 'resource2': {'type': 'OS::Heat::ResourceGroup', 'properties': {'resource_def': {'type': egg_url}, 'with': {'get_file': ham_url}}}}}, json.loads(files.get(template_url)))
        self.assertEqual(yaml.safe_load(self.foo_template.decode('utf-8')), json.loads(files.get(foo_url)))
        self.assertEqual({'heat_template_version': '2013-05-23', 'parameters': {'param1': {'type': 'string'}}, 'resources': {'resource1': {'properties': {'foo': 'bar'}, 'type': one_url}, 'resource2': {'type': 'OS::Heat::ResourceGroup', 'properties': {'resource_def': {'type': two_url}, 'with': {'get_file': three_url}}}}}, json.loads(files.get(egg_url)))
        self.assertEqual(b'ham contents', files.get(ham_url))
        self.assertEqual(yaml.safe_load(self.foo_template.decode('utf-8')), json.loads(files.get(one_url)))
        self.assertEqual(yaml.safe_load(self.foo_template.decode('utf-8')), json.loads(files.get(two_url)))
        self.assertEqual(b'three contents', files.get(three_url))
        mock_url.assert_has_calls([mock.call(env_url), mock.call(template_url), mock.call(foo_url), mock.call(egg_url), mock.call(ham_url), mock.call(one_url), mock.call(two_url), mock.call(three_url)], any_order=True)