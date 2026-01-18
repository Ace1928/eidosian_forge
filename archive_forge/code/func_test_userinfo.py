from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_userinfo(self) -> None:
    """
        L{URL.fromText} will parse the C{userinfo} portion of the URI
        separately from the host and port.
        """
    url = URL.fromText('http://someuser:somepassword@example.com/some-segment@ignore')
    self.assertEqual(url.authority(True), 'someuser:somepassword@example.com')
    self.assertEqual(url.authority(False), 'someuser:@example.com')
    self.assertEqual(url.userinfo, 'someuser:somepassword')
    self.assertEqual(url.user, 'someuser')
    self.assertEqual(url.asText(), 'http://someuser:@example.com/some-segment@ignore')
    self.assertEqual(url.replace(userinfo='someuser').asText(), 'http://someuser@example.com/some-segment@ignore')