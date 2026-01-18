from heat.common import identifier
from heat.tests import common
def test_url_tenant_decode(self):
    enc_arn = 'arn%3Aopenstack%3Aheat%3A%3A%253A%252F%3Astacks%2Fs%2Fi'
    url = self.url_prefix + enc_arn
    hi = identifier.HeatIdentifier.from_arn_url(url)
    self.assertEqual(':/', hi.tenant)