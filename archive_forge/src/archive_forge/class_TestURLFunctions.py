import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
class TestURLFunctions(testtools.TestCase):

    def setUp(self):
        super(TestURLFunctions, self).setUp()
        self.m = mock.MagicMock()
        self.addCleanup(self.m.UnsetStubs)

    def test_normalise_file_path_to_url_relative(self):
        self.assertEqual('file://' + request.pathname2url('%s/foo' % os.getcwd()), utils.normalise_file_path_to_url('foo'))

    def test_normalise_file_path_to_url_absolute(self):
        self.assertEqual('file:///tmp/foo', utils.normalise_file_path_to_url('/tmp/foo'))

    def test_normalise_file_path_to_url_file(self):
        self.assertEqual('file:///tmp/foo', utils.normalise_file_path_to_url('file:///tmp/foo'))

    def test_normalise_file_path_to_url_http(self):
        self.assertEqual('http://localhost/foo', utils.normalise_file_path_to_url('http://localhost/foo'))

    def test_get_template_url(self):
        tmpl_file = '/opt/stack/template.yaml'
        tmpl_url = 'file:///opt/stack/template.yaml'
        self.assertEqual(utils.get_template_url(tmpl_file, None), tmpl_url)
        self.assertEqual(utils.get_template_url(None, tmpl_url), tmpl_url)
        self.assertIsNone(utils.get_template_url(None, None))

    def test_base_url_for_url(self):
        self.assertEqual('file:///foo/bar', utils.base_url_for_url('file:///foo/bar/baz'))
        self.assertEqual('file:///foo/bar', utils.base_url_for_url('file:///foo/bar/baz.txt'))
        self.assertEqual('file:///foo/bar', utils.base_url_for_url('file:///foo/bar/'))
        self.assertEqual('file:///', utils.base_url_for_url('file:///'))
        self.assertEqual('file:///', utils.base_url_for_url('file:///foo'))
        self.assertEqual('http://foo/bar', utils.base_url_for_url('http://foo/bar/'))
        self.assertEqual('http://foo/bar', utils.base_url_for_url('http://foo/bar/baz.template'))