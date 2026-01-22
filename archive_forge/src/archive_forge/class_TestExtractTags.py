from heat.common import param_utils
from heat.tests import common
class TestExtractTags(common.HeatTestCase):

    def test_extract_tags(self):
        self.assertRaises(ValueError, param_utils.extract_tags, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,a')
        self.assertEqual(['foo', 'bar'], param_utils.extract_tags('foo,bar'))