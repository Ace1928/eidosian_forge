import unittest
from boto.s3.cors import CORSConfiguration
def test_one_rule_with_id(self):
    cfg = CORSConfiguration()
    cfg.add_rule(['PUT', 'POST', 'DELETE'], 'http://www.example.com', allowed_header='*', max_age_seconds=3000, expose_header='x-amz-server-side-encryption', id='foobar_rule')
    self.assertEqual(cfg.to_xml(), CORS_BODY_1)