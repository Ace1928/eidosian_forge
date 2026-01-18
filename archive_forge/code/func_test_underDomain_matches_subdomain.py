from collections import defaultdict
import unittest
from lazr.uri import (
def test_underDomain_matches_subdomain(self):
    uri = URI('http://code.launchpad.dev/foo')
    self.assertTrue(uri.underDomain('code.launchpad.dev'))
    self.assertTrue(uri.underDomain('launchpad.dev'))
    self.assertTrue(uri.underDomain(''))