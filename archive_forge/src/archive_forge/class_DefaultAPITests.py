import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
@skipIf(not FilePath('/dev/tty').exists(), 'Platform lacks /dev/tty')
class DefaultAPITests(TestCase):
    """
    The API in L{twisted.conch.client.default.verifyHostKey} is the integration
    point between the code in the rest of conch and L{KnownHostsFile}.
    """

    def patchedOpen(self, fname, mode, **kwargs):
        """
        The patched version of 'open'; this returns a L{FakeFile} that the
        instantiated L{ConsoleUI} can use.
        """
        self.assertEqual(fname, '/dev/tty')
        self.assertEqual(mode, 'r+b')
        self.assertEqual(kwargs['buffering'], 0)
        return self.fakeFile

    def setUp(self):
        """
        Patch 'open' in verifyHostKey.
        """
        self.fakeFile = FakeFile()
        self.patch(default, '_open', self.patchedOpen)
        self.hostsOption = self.mktemp()
        self.hashedEntries = {}
        knownHostsFile = KnownHostsFile(FilePath(self.hostsOption))
        for host in (b'exists.example.com', b'4.3.2.1'):
            entry = knownHostsFile.addHostKey(host, Key.fromString(sampleKey))
            self.hashedEntries[host] = entry
        knownHostsFile.save()
        self.fakeTransport = FakeObject()
        self.fakeTransport.factory = FakeObject()
        self.options = self.fakeTransport.factory.options = {'host': b'exists.example.com', 'known-hosts': self.hostsOption}

    def test_verifyOKKey(self):
        """
        L{default.verifyHostKey} should return a L{Deferred} which fires with
        C{1} when passed a host, IP, and key which already match the
        known_hosts file it is supposed to check.
        """
        l = []
        default.verifyHostKey(self.fakeTransport, b'4.3.2.1', sampleKey, b"I don't care.").addCallback(l.append)
        self.assertEqual([1], l)

    def replaceHome(self, tempHome):
        """
        Replace the HOME environment variable until the end of the current
        test, with the given new home-directory, so that L{os.path.expanduser}
        will yield controllable, predictable results.

        @param tempHome: the pathname to replace the HOME variable with.

        @type tempHome: L{str}
        """
        oldHome = os.environ.get('HOME')

        def cleanupHome():
            if oldHome is None:
                del os.environ['HOME']
            else:
                os.environ['HOME'] = oldHome
        self.addCleanup(cleanupHome)
        os.environ['HOME'] = tempHome

    def test_noKnownHostsOption(self):
        """
        L{default.verifyHostKey} should find your known_hosts file in
        ~/.ssh/known_hosts if you don't specify one explicitly on the command
        line.
        """
        l = []
        tmpdir = self.mktemp()
        oldHostsOption = self.hostsOption
        hostsNonOption = FilePath(tmpdir).child('.ssh').child('known_hosts')
        hostsNonOption.parent().makedirs()
        FilePath(oldHostsOption).moveTo(hostsNonOption)
        self.replaceHome(tmpdir)
        self.options['known-hosts'] = None
        default.verifyHostKey(self.fakeTransport, b'4.3.2.1', sampleKey, b"I don't care.").addCallback(l.append)
        self.assertEqual([1], l)

    def test_verifyHostButNotIP(self):
        """
        L{default.verifyHostKey} should return a L{Deferred} which fires with
        C{1} when passed a host which matches with an IP is not present in its
        known_hosts file, and should also warn the user that it has added the
        IP address.
        """
        l = []
        default.verifyHostKey(self.fakeTransport, b'8.7.6.5', sampleKey, b'Fingerprint not required.').addCallback(l.append)
        self.assertEqual(["Warning: Permanently added the RSA host key for IP address '8.7.6.5' to the list of known hosts."], self.fakeFile.outchunks)
        self.assertEqual([1], l)
        knownHostsFile = KnownHostsFile.fromPath(FilePath(self.hostsOption))
        self.assertTrue(knownHostsFile.hasHostKey(b'8.7.6.5', Key.fromString(sampleKey)))

    def test_verifyQuestion(self):
        """
        L{default.verifyHostKey} should return a L{Default} which fires with
        C{0} when passed an unknown host that the user refuses to acknowledge.
        """
        self.fakeTransport.factory.options['host'] = b'fake.example.com'
        self.fakeFile.inlines.append(b'no')
        d = default.verifyHostKey(self.fakeTransport, b'9.8.7.6', otherSampleKey, b'No fingerprint!')
        self.assertEqual([b"The authenticity of host 'fake.example.com (9.8.7.6)' can't be established.\nRSA key fingerprint is SHA256:vD0YydsNIUYJa7yLZl3tIL8h0vZvQ8G+HPG7JLmQV0s=.\nAre you sure you want to continue connecting (yes/no)? "], self.fakeFile.outchunks)
        return self.assertFailure(d, UserRejectedKey)

    def test_verifyBadKey(self):
        """
        L{default.verifyHostKey} should return a L{Deferred} which fails with
        L{HostKeyChanged} if the host key is incorrect.
        """
        d = default.verifyHostKey(self.fakeTransport, b'4.3.2.1', otherSampleKey, 'Again, not required.')
        return self.assertFailure(d, HostKeyChanged)

    def test_inKnownHosts(self):
        """
        L{default.isInKnownHosts} should return C{1} when a host with a key
        is in the known hosts file.
        """
        host = self.hashedEntries[b'4.3.2.1'].toString().split()[0]
        r = default.isInKnownHosts(host, Key.fromString(sampleKey).blob(), {'known-hosts': FilePath(self.hostsOption).path})
        self.assertEqual(1, r)

    def test_notInKnownHosts(self):
        """
        L{default.isInKnownHosts} should return C{0} when a host with a key
        is not in the known hosts file.
        """
        r = default.isInKnownHosts('not.there', b'irrelevant', {'known-hosts': FilePath(self.hostsOption).path})
        self.assertEqual(0, r)

    def test_inKnownHostsKeyChanged(self):
        """
        L{default.isInKnownHosts} should return C{2} when a host with a key
        other than the given one is in the known hosts file.
        """
        host = self.hashedEntries[b'4.3.2.1'].toString().split()[0]
        r = default.isInKnownHosts(host, Key.fromString(otherSampleKey).blob(), {'known-hosts': FilePath(self.hostsOption).path})
        self.assertEqual(2, r)