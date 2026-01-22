import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
class TestCompareElements(base.BaseTestCase):

    def test_compare_elements(self):
        self.assertFalse(helpers.compare_elements([], ['napoli']))
        self.assertFalse(helpers.compare_elements(None, ['napoli']))
        self.assertFalse(helpers.compare_elements(['napoli'], []))
        self.assertFalse(helpers.compare_elements(['napoli'], None))
        self.assertFalse(helpers.compare_elements(['napoli', 'juve'], ['juve']))
        self.assertTrue(helpers.compare_elements(['napoli', 'juve'], ['napoli', 'juve']))
        self.assertTrue(helpers.compare_elements(['napoli', 'juve'], ['juve', 'napoli']))