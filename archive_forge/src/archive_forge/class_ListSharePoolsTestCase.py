from manilaclient.tests.functional.osc import base
class ListSharePoolsTestCase(base.OSCClientTestBase):

    def test_limits_show_absolute(self):
        limits = self.listing_result('share', ' limits show --absolute')
        self.assertTableStruct(limits, ['Name', 'Value'])

    def test_limits_show_rate(self):
        limits = self.listing_result('share', ' limits show --rate --print-empty')
        self.assertTableStruct(limits, ['Verb', 'Regex', 'URI', 'Value', 'Remaining', 'Unit', 'Next Available'])