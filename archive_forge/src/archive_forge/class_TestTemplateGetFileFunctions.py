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
class TestTemplateGetFileFunctions(testtools.TestCase):
    hot_template = b"heat_template_version: 2013-05-23\nresources:\n  resource1:\n    type: OS::type1\n    properties:\n      foo: {get_file: foo.yaml}\n      bar:\n        get_file:\n          'http://localhost/bar.yaml'\n  resource2:\n    type: OS::type1\n    properties:\n      baz:\n      - {get_file: baz/baz1.yaml}\n      - {get_file: baz/baz2.yaml}\n      - {get_file: baz/baz3.yaml}\n      ignored_list: {get_file: [ignore, me]}\n      ignored_dict: {get_file: {ignore: me}}\n      ignored_none: {get_file: }\n    "

    def test_hot_template(self, mock_url):
        tmpl_file = '/home/my/dir/template.yaml'
        url = 'file:///home/my/dir/template.yaml'
        mock_url.side_effect = [io.BytesIO(self.hot_template), io.BytesIO(b'bar contents'), io.BytesIO(b'foo contents'), io.BytesIO(b'baz1 contents'), io.BytesIO(b'baz2 contents'), io.BytesIO(b'baz3 contents')]
        files, tmpl_parsed = template_utils.get_template_contents(template_file=tmpl_file)
        self.assertEqual({'heat_template_version': '2013-05-23', 'resources': {'resource1': {'type': 'OS::type1', 'properties': {'bar': {'get_file': 'http://localhost/bar.yaml'}, 'foo': {'get_file': 'file:///home/my/dir/foo.yaml'}}}, 'resource2': {'type': 'OS::type1', 'properties': {'baz': [{'get_file': 'file:///home/my/dir/baz/baz1.yaml'}, {'get_file': 'file:///home/my/dir/baz/baz2.yaml'}, {'get_file': 'file:///home/my/dir/baz/baz3.yaml'}], 'ignored_list': {'get_file': ['ignore', 'me']}, 'ignored_dict': {'get_file': {'ignore': 'me'}}, 'ignored_none': {'get_file': None}}}}}, tmpl_parsed)
        mock_url.assert_has_calls([mock.call(url), mock.call('http://localhost/bar.yaml'), mock.call('file:///home/my/dir/foo.yaml'), mock.call('file:///home/my/dir/baz/baz1.yaml'), mock.call('file:///home/my/dir/baz/baz2.yaml'), mock.call('file:///home/my/dir/baz/baz3.yaml')], any_order=True)

    def test_hot_template_outputs(self, mock_url):
        tmpl_file = '/home/my/dir/template.yaml'
        url = 'file://%s' % tmpl_file
        foo_url = 'file:///home/my/dir/foo.yaml'
        contents = b'\nheat_template_version: 2013-05-23\noutputs:\n  contents:\n    value:\n      get_file: foo.yaml\n'
        mock_url.side_effect = [io.BytesIO(contents), io.BytesIO(b'foo contents')]
        files = template_utils.get_template_contents(template_file=tmpl_file)[0]
        self.assertEqual({foo_url: b'foo contents'}, files)
        mock_url.assert_has_calls([mock.call(url), mock.call(foo_url)])

    def test_hot_template_same_file(self, mock_url):
        tmpl_file = '/home/my/dir/template.yaml'
        url = 'file://%s' % tmpl_file
        foo_url = 'file:///home/my/dir/foo.yaml'
        contents = b'\nheat_template_version: 2013-05-23\n\noutputs:\n  contents:\n    value:\n      get_file: foo.yaml\n  template:\n    value:\n      get_file: foo.yaml\n'
        mock_url.side_effect = [io.BytesIO(contents), io.BytesIO(b'foo contents')]
        files = template_utils.get_template_contents(template_file=tmpl_file)[0]
        self.assertEqual({foo_url: b'foo contents'}, files)
        mock_url.assert_has_calls([mock.call(url), mock.call(foo_url)])