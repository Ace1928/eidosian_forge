from unittest import mock
import testtools
from urllib import parse
from heatclient.common import utils
from heatclient.v1 import resources
class ResourceStackNameTest(testtools.TestCase):

    def test_stack_name(self):
        resource = resources.Resource(None, {'links': [{'href': 'http://heat.example.com:8004/foo/12/resources/foobar', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo/12', 'rel': 'stack'}]})
        self.assertEqual('foo', resource.stack_name)

    def test_stack_name_no_links(self):
        resource = resources.Resource(None, {})
        self.assertIsNone(resource.stack_name)