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
class TestGetTemplateContents(testtools.TestCase):

    def test_get_template_contents_file(self):
        with tempfile.NamedTemporaryFile() as tmpl_file:
            tmpl = b'{"AWSTemplateFormatVersion" : "2010-09-09", "foo": "bar"}'
            tmpl_file.write(tmpl)
            tmpl_file.flush()
            files, tmpl_parsed = template_utils.get_template_contents(tmpl_file.name)
            self.assertEqual({'AWSTemplateFormatVersion': '2010-09-09', 'foo': 'bar'}, tmpl_parsed)
            self.assertEqual({}, files)

    def test_get_template_contents_file_empty(self):
        with tempfile.NamedTemporaryFile() as tmpl_file:
            ex = self.assertRaises(exc.CommandError, template_utils.get_template_contents, tmpl_file.name)
            self.assertEqual('Could not fetch template from file://%s' % tmpl_file.name, str(ex))

    def test_get_template_file_nonextant(self):
        nonextant_file = '/template/dummy/file/path/and/name.yaml'
        ex = self.assertRaises(error.URLError, template_utils.get_template_contents, nonextant_file)
        self.assertEqual("<urlopen error [Errno 2] No such file or directory: '%s'>" % nonextant_file, str(ex))

    def test_get_template_contents_file_none(self):
        ex = self.assertRaises(exc.CommandError, template_utils.get_template_contents)
        self.assertEqual('Need to specify exactly one of [--template-file, --template-url or --template-object] or --existing', str(ex))

    def test_get_template_contents_file_none_existing(self):
        files, tmpl_parsed = template_utils.get_template_contents(existing=True)
        self.assertIsNone(tmpl_parsed)
        self.assertEqual({}, files)

    def test_get_template_contents_parse_error(self):
        with tempfile.NamedTemporaryFile() as tmpl_file:
            tmpl = b'{"foo": "bar"'
            tmpl_file.write(tmpl)
            tmpl_file.flush()
            ex = self.assertRaises(exc.CommandError, template_utils.get_template_contents, tmpl_file.name)
            self.assertThat(str(ex), matchers.MatchesRegex('Error parsing template file://%s ' % tmpl_file.name))

    @mock.patch('urllib.request.urlopen')
    def test_get_template_contents_url(self, mock_url):
        tmpl = b'{"AWSTemplateFormatVersion" : "2010-09-09", "foo": "bar"}'
        url = 'http://no.where/path/to/a.yaml'
        mock_url.return_value = io.BytesIO(tmpl)
        files, tmpl_parsed = template_utils.get_template_contents(template_url=url)
        self.assertEqual({'AWSTemplateFormatVersion': '2010-09-09', 'foo': 'bar'}, tmpl_parsed)
        self.assertEqual({}, files)
        mock_url.assert_called_with(url)

    def test_get_template_contents_object(self):
        tmpl = '{"AWSTemplateFormatVersion" : "2010-09-09", "foo": "bar"}'
        url = 'http://no.where/path/to/a.yaml'
        self.object_requested = False

        def object_request(method, object_url):
            self.object_requested = True
            self.assertEqual('GET', method)
            self.assertEqual('http://no.where/path/to/a.yaml', object_url)
            return tmpl
        files, tmpl_parsed = template_utils.get_template_contents(template_object=url, object_request=object_request)
        self.assertEqual({'AWSTemplateFormatVersion': '2010-09-09', 'foo': 'bar'}, tmpl_parsed)
        self.assertEqual({}, files)
        self.assertTrue(self.object_requested)

    def test_get_nested_stack_template_contents_object(self):
        tmpl = '{"heat_template_version": "2016-04-08","resources": {"FooBar": {"type": "foo/bar.yaml"}}}'
        url = 'http://no.where/path/to/a.yaml'
        self.object_requested = False

        def object_request(method, object_url):
            self.object_requested = True
            self.assertEqual('GET', method)
            self.assertTrue(object_url.startswith('http://no.where/path/to/'))
            if object_url == url:
                return tmpl
            else:
                return '{"heat_template_version": "2016-04-08"}'
        files, tmpl_parsed = template_utils.get_template_contents(template_object=url, object_request=object_request)
        self.assertEqual(files['http://no.where/path/to/foo/bar.yaml'], '{"heat_template_version": "2016-04-08"}')
        self.assertTrue(self.object_requested)

    def check_non_utf8_content(self, filename, content):
        base_url = 'file:///tmp'
        url = '%s/%s' % (base_url, filename)
        template = {'resources': {'one_init': {'type': 'OS::Heat::CloudConfig', 'properties': {'cloud_config': {'write_files': [{'path': '/tmp/%s' % filename, 'content': {'get_file': url}, 'encoding': 'b64'}]}}}}}
        with mock.patch('urllib.request.urlopen') as mock_url:
            raw_content = base64.decode_as_bytes(content)
            response = io.BytesIO(raw_content)
            mock_url.return_value = response
            files = {}
            template_utils.resolve_template_get_files(template, files, base_url)
            self.assertEqual({url: content}, files)
            mock_url.assert_called_with(url)

    def test_get_zip_content(self):
        filename = 'heat.zip'
        content = b'UEsDBAoAAAAAAEZZWkRbOAuBBQAAAAUAAAAIABwAaGVhdC50eHRVVAkAAxRbDVNYht9SdXgLAAEE\n6AMAAATpAwAAaGVhdApQSwECHgMKAAAAAABGWVpEWzgLgQUAAAAFAAAACAAYAAAAAAABAAAApIEA\nAAAAaGVhdC50eHRVVAUAAxRbDVN1eAsAAQToAwAABOkDAABQSwUGAAAAAAEAAQBOAAAARwAAAAAA\n'
        self.assertIn(b'\x00', base64.decode_as_bytes(content))
        decoded_content = base64.decode_as_bytes(content)
        self.assertRaises(UnicodeDecodeError, decoded_content.decode)
        self.check_non_utf8_content(filename=filename, content=content)

    def test_get_utf16_content(self):
        filename = 'heat.utf16'
        content = b'//4tTkhTCgA=\n'
        self.assertIn(b'\x00', base64.decode_as_bytes(content))
        decoded_content = base64.decode_as_bytes(content)
        self.assertRaises(UnicodeDecodeError, decoded_content.decode)
        self.check_non_utf8_content(filename=filename, content=content)

    def test_get_gb18030_content(self):
        filename = 'heat.gb18030'
        content = b'1tDO5wo=\n'
        self.assertNotIn('\x00', base64.decode_as_bytes(content))
        decoded_content = base64.decode_as_bytes(content)
        self.assertRaises(UnicodeDecodeError, decoded_content.decode)
        self.check_non_utf8_content(filename=filename, content=content)