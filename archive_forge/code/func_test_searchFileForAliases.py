from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_searchFileForAliases(self) -> None:
    """
        For a host with a canonical name and one or more aliases,
        L{searchFileFor} can find an address given any of the names.
        """
    hosts = self.path()
    hosts.setContent(b'127.0.1.1\thelmut.example.org\thelmut\n# a comment\n::1 localhost ip6-localhost ip6-loopback\n')
    self.assertEqual(searchFileFor(hosts.path, b'helmut'), '127.0.1.1')
    self.assertEqual(searchFileFor(hosts.path, b'helmut.example.org'), '127.0.1.1')
    self.assertEqual(searchFileFor(hosts.path, b'ip6-localhost'), '::1')
    self.assertEqual(searchFileFor(hosts.path, b'ip6-loopback'), '::1')
    self.assertEqual(searchFileFor(hosts.path, b'localhost'), '::1')