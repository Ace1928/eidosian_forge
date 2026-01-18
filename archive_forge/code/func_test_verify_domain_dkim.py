from tests.unit import unittest
from boto.ses.connection import SESConnection
from boto.ses import exceptions
def test_verify_domain_dkim(self):
    with self.assertRaises(exceptions.SESDomainNotConfirmedError):
        self.ses.verify_domain_dkim('example.com')