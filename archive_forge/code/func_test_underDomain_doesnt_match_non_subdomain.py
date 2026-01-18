from collections import defaultdict
import unittest
from lazr.uri import (
def test_underDomain_doesnt_match_non_subdomain(self):
    uri = URI('http://code.launchpad.dev/foo')
    self.assertFalse(uri.underDomain('beta.code.launchpad.dev'))
    self.assertFalse(uri.underDomain('google.com'))
    self.assertFalse(uri.underDomain('unchpad.dev'))