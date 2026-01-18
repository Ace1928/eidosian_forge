from cinderclient.tests.functional import base
def test_rate_limits(self):
    rate_limits = self.cinder('rate-limits')
    self.assertTableHeaders(rate_limits, ['Verb', 'URI', 'Value', 'Remain', 'Unit', 'Next_Available'])