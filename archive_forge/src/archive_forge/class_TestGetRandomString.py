import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
class TestGetRandomString(base.BaseTestCase):

    def test_get_random_string(self):
        length = 127
        random_string = helpers.get_random_string(length)
        self.assertEqual(length, len(random_string))
        regex = re.compile('^[0-9a-fA-F]+$')
        self.assertIsNotNone(regex.match(random_string))