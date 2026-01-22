import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
class PrepareQueryStringTestCase(test_utils.TestCase):

    def setUp(self):
        super(PrepareQueryStringTestCase, self).setUp()
        self.ustr = b'?\xd0\xbf=1&\xd1\x80=2'
        self.ustr = self.ustr.decode('utf8')
        self.cases = (({}, ''), (None, ''), ({'2': 2, '10': 1}, '?10=1&2=2'), ({'abc': 1, 'abc1': 2}, '?abc=1&abc1=2'), ({b'\xd0\xbf': 1, b'\xd1\x80': 2}, self.ustr), ({(1, 2): '1', (3, 4): '2'}, '?(1, 2)=1&(3, 4)=2'))

    def test_convert_dict_to_string(self):
        for case in self.cases:
            self.assertEqual(case[1], parse.unquote_plus(utils.prepare_query_string(case[0])))

    def test_get_url_with_filter(self):
        url = '/fake'
        for case in self.cases:
            self.assertEqual('%s%s' % (url, case[1]), parse.unquote_plus(utils.get_url_with_filter(url, case[0])))