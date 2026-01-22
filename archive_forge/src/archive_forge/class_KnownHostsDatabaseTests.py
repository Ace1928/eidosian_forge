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
class KnownHostsDatabaseTests(TestCase):
    """
    Tests for L{KnownHostsFile}.
    """

    def pathWithContent(self, content):
        """
        Return a FilePath with the given initial content.
        """
        fp = FilePath(self.mktemp())
        fp.setContent(content)
        return fp

    def loadSampleHostsFile(self, content=sampleHashedLine + otherSamplePlaintextLine + b'\n# That was a blank line.\nThis is just unparseable.\n|1|This also unparseable.\n'):
        """
        Return a sample hosts file, with keys for www.twistedmatrix.com and
        divmod.com present.
        """
        return KnownHostsFile.fromPath(self.pathWithContent(content))

    def test_readOnlySavePath(self):
        """
        L{KnownHostsFile.savePath} is read-only; if an assignment is made to
        it, L{AttributeError} is raised and the value is unchanged.
        """
        path = FilePath(self.mktemp())
        new = FilePath(self.mktemp())
        hostsFile = KnownHostsFile(path)
        self.assertRaises(AttributeError, setattr, hostsFile, 'savePath', new)
        self.assertEqual(path, hostsFile.savePath)

    def test_defaultInitializerIgnoresExisting(self):
        """
        The default initializer for L{KnownHostsFile} disregards any existing
        contents in the save path.
        """
        hostsFile = KnownHostsFile(self.pathWithContent(sampleHashedLine))
        self.assertEqual([], list(hostsFile.iterentries()))

    def test_defaultInitializerClobbersExisting(self):
        """
        After using the default initializer for L{KnownHostsFile}, the first use
        of L{KnownHostsFile.save} overwrites any existing contents in the save
        path.
        """
        path = self.pathWithContent(sampleHashedLine)
        hostsFile = KnownHostsFile(path)
        entry = hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
        hostsFile.save()
        self.assertEqual([entry], list(hostsFile.iterentries()))
        self.assertEqual(entry.toString() + b'\n', path.getContent())

    def test_saveResetsClobberState(self):
        """
        After L{KnownHostsFile.save} is used once with an instance initialized
        by the default initializer, contents of the save path are respected and
        preserved.
        """
        hostsFile = KnownHostsFile(self.pathWithContent(sampleHashedLine))
        preSave = hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
        hostsFile.save()
        postSave = hostsFile.addHostKey(b'another.example.com', Key.fromString(thirdSampleKey))
        hostsFile.save()
        self.assertEqual([preSave, postSave], list(hostsFile.iterentries()))

    def test_loadFromPath(self):
        """
        Loading a L{KnownHostsFile} from a path with six entries in it will
        result in a L{KnownHostsFile} object with six L{IKnownHostEntry}
        providers in it.
        """
        hostsFile = self.loadSampleHostsFile()
        self.assertEqual(6, len(list(hostsFile.iterentries())))

    def test_iterentriesUnsaved(self):
        """
        If the save path for a L{KnownHostsFile} does not exist,
        L{KnownHostsFile.iterentries} still returns added but unsaved entries.
        """
        hostsFile = KnownHostsFile(FilePath(self.mktemp()))
        hostsFile.addHostKey(b'www.example.com', Key.fromString(sampleKey))
        self.assertEqual(1, len(list(hostsFile.iterentries())))

    def test_verifyHashedEntry(self):
        """
        Loading a L{KnownHostsFile} from a path containing a single valid
        L{HashedEntry} entry will result in a L{KnownHostsFile} object
        with one L{IKnownHostEntry} provider.
        """
        hostsFile = self.loadSampleHostsFile(sampleHashedLine)
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], HashedEntry)
        self.assertTrue(entries[0].matchesHost(b'www.twistedmatrix.com'))
        self.assertEqual(1, len(entries))

    def test_verifyPlainEntry(self):
        """
        Loading a L{KnownHostsFile} from a path containing a single valid
        L{PlainEntry} entry will result in a L{KnownHostsFile} object
        with one L{IKnownHostEntry} provider.
        """
        hostsFile = self.loadSampleHostsFile(otherSamplePlaintextLine)
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], PlainEntry)
        self.assertTrue(entries[0].matchesHost(b'divmod.com'))
        self.assertEqual(1, len(entries))

    def test_verifyUnparsedEntry(self):
        """
        Loading a L{KnownHostsFile} from a path that only contains '
' will
        result in a L{KnownHostsFile} object containing a L{UnparsedEntry}
        object.
        """
        hostsFile = self.loadSampleHostsFile(b'\n')
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], UnparsedEntry)
        self.assertEqual(entries[0].toString(), b'')
        self.assertEqual(1, len(entries))

    def test_verifyUnparsedComment(self):
        """
        Loading a L{KnownHostsFile} from a path that contains a comment will
        result in a L{KnownHostsFile} object containing a L{UnparsedEntry}
        object.
        """
        hostsFile = self.loadSampleHostsFile(b'# That was a blank line.\n')
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], UnparsedEntry)
        self.assertEqual(entries[0].toString(), b'# That was a blank line.')

    def test_verifyUnparsableLine(self):
        """
        Loading a L{KnownHostsFile} from a path that contains an unparseable
        line will be represented as an L{UnparsedEntry} instance.
        """
        hostsFile = self.loadSampleHostsFile(b'This is just unparseable.\n')
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], UnparsedEntry)
        self.assertEqual(entries[0].toString(), b'This is just unparseable.')
        self.assertEqual(1, len(entries))

    def test_verifyUnparsableEncryptionMarker(self):
        """
        Loading a L{KnownHostsFile} from a path containing an unparseable line
        that starts with an encryption marker will be represented as an
        L{UnparsedEntry} instance.
        """
        hostsFile = self.loadSampleHostsFile(b'|1|This is unparseable.\n')
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], UnparsedEntry)
        self.assertEqual(entries[0].toString(), b'|1|This is unparseable.')
        self.assertEqual(1, len(entries))

    def test_loadNonExistent(self):
        """
        Loading a L{KnownHostsFile} from a path that does not exist should
        result in an empty L{KnownHostsFile} that will save back to that path.
        """
        pn = self.mktemp()
        knownHostsFile = KnownHostsFile.fromPath(FilePath(pn))
        entries = list(knownHostsFile.iterentries())
        self.assertEqual([], entries)
        self.assertFalse(FilePath(pn).exists())
        knownHostsFile.save()
        self.assertTrue(FilePath(pn).exists())

    def test_loadNonExistentParent(self):
        """
        Loading a L{KnownHostsFile} from a path whose parent directory does not
        exist should result in an empty L{KnownHostsFile} that will save back
        to that path, creating its parent directory(ies) in the process.
        """
        thePath = FilePath(self.mktemp())
        knownHostsPath = thePath.child('foo').child(b'known_hosts')
        knownHostsFile = KnownHostsFile.fromPath(knownHostsPath)
        knownHostsFile.save()
        knownHostsPath.restat(False)
        self.assertTrue(knownHostsPath.exists())

    def test_savingAddsEntry(self):
        """
        L{KnownHostsFile.save} will write out a new file with any entries
        that have been added.
        """
        path = self.pathWithContent(sampleHashedLine + otherSamplePlaintextLine)
        knownHostsFile = KnownHostsFile.fromPath(path)
        newEntry = knownHostsFile.addHostKey(b'some.example.com', Key.fromString(thirdSampleKey))
        expectedContent = sampleHashedLine + otherSamplePlaintextLine + HashedEntry.MAGIC + b2a_base64(newEntry._hostSalt).strip() + b'|' + b2a_base64(newEntry._hostHash).strip() + b' ssh-rsa ' + thirdSampleEncodedKey + b'\n'
        self.assertEqual(3, expectedContent.count(b'\n'))
        knownHostsFile.save()
        self.assertEqual(expectedContent, path.getContent())

    def test_savingAvoidsDuplication(self):
        """
        L{KnownHostsFile.save} only writes new entries to the save path, not
        entries which were added and already written by a previous call to
        C{save}.
        """
        path = FilePath(self.mktemp())
        knownHosts = KnownHostsFile(path)
        entry = knownHosts.addHostKey(b'some.example.com', Key.fromString(sampleKey))
        knownHosts.save()
        knownHosts.save()
        knownHosts = KnownHostsFile.fromPath(path)
        self.assertEqual([entry], list(knownHosts.iterentries()))

    def test_savingsPreservesExisting(self):
        """
        L{KnownHostsFile.save} will not overwrite existing entries in its save
        path, even if they were only added after the L{KnownHostsFile} instance
        was initialized.
        """
        path = self.pathWithContent(sampleHashedLine)
        knownHosts = KnownHostsFile.fromPath(path)
        with path.open('a') as hostsFileObj:
            hostsFileObj.write(otherSamplePlaintextLine)
        key = Key.fromString(thirdSampleKey)
        knownHosts.addHostKey(b'brandnew.example.com', key)
        knownHosts.save()
        knownHosts = KnownHostsFile.fromPath(path)
        self.assertEqual([True, True, True], [knownHosts.hasHostKey(b'www.twistedmatrix.com', Key.fromString(sampleKey)), knownHosts.hasHostKey(b'divmod.com', Key.fromString(otherSampleKey)), knownHosts.hasHostKey(b'brandnew.example.com', key)])

    def test_hasPresentKey(self):
        """
        L{KnownHostsFile.hasHostKey} returns C{True} when a key for the given
        hostname is present and matches the expected key.
        """
        hostsFile = self.loadSampleHostsFile()
        self.assertTrue(hostsFile.hasHostKey(b'www.twistedmatrix.com', Key.fromString(sampleKey)))

    def test_notPresentKey(self):
        """
        L{KnownHostsFile.hasHostKey} returns C{False} when a key for the given
        hostname is not present.
        """
        hostsFile = self.loadSampleHostsFile()
        self.assertFalse(hostsFile.hasHostKey(b'non-existent.example.com', Key.fromString(sampleKey)))
        self.assertTrue(hostsFile.hasHostKey(b'www.twistedmatrix.com', Key.fromString(sampleKey)))
        self.assertFalse(hostsFile.hasHostKey(b'www.twistedmatrix.com', Key.fromString(ecdsaSampleKey)))

    def test_hasLaterAddedKey(self):
        """
        L{KnownHostsFile.hasHostKey} returns C{True} when a key for the given
        hostname is present in the file, even if it is only added to the file
        after the L{KnownHostsFile} instance is initialized.
        """
        key = Key.fromString(sampleKey)
        entry = PlainEntry([b'brandnew.example.com'], key.sshType(), key, b'')
        hostsFile = self.loadSampleHostsFile()
        with hostsFile.savePath.open('a') as hostsFileObj:
            hostsFileObj.write(entry.toString() + b'\n')
        self.assertEqual(True, hostsFile.hasHostKey(b'brandnew.example.com', key))

    def test_savedEntryHasKeyMismatch(self):
        """
        L{KnownHostsFile.hasHostKey} raises L{HostKeyChanged} if the host key is
        present in the underlying file, but different from the expected one.
        The resulting exception should have an C{offendingEntry} indicating the
        given entry.
        """
        hostsFile = self.loadSampleHostsFile()
        entries = list(hostsFile.iterentries())
        exception = self.assertRaises(HostKeyChanged, hostsFile.hasHostKey, b'www.twistedmatrix.com', Key.fromString(otherSampleKey))
        self.assertEqual(exception.offendingEntry, entries[0])
        self.assertEqual(exception.lineno, 1)
        self.assertEqual(exception.path, hostsFile.savePath)

    def test_savedEntryAfterAddHasKeyMismatch(self):
        """
        Even after a new entry has been added in memory but not yet saved, the
        L{HostKeyChanged} exception raised by L{KnownHostsFile.hasHostKey} has a
        C{lineno} attribute which indicates the 1-based line number of the
        offending entry in the underlying file when the given host key does not
        match the expected host key.
        """
        hostsFile = self.loadSampleHostsFile()
        hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
        exception = self.assertRaises(HostKeyChanged, hostsFile.hasHostKey, b'www.twistedmatrix.com', Key.fromString(otherSampleKey))
        self.assertEqual(exception.lineno, 1)
        self.assertEqual(exception.path, hostsFile.savePath)

    def test_unsavedEntryHasKeyMismatch(self):
        """
        L{KnownHostsFile.hasHostKey} raises L{HostKeyChanged} if the host key is
        present in memory (but not yet saved), but different from the expected
        one.  The resulting exception has a C{offendingEntry} indicating the
        given entry, but no filename or line number information (reflecting the
        fact that the entry exists only in memory).
        """
        hostsFile = KnownHostsFile(FilePath(self.mktemp()))
        entry = hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
        exception = self.assertRaises(HostKeyChanged, hostsFile.hasHostKey, b'www.example.com', Key.fromString(thirdSampleKey))
        self.assertEqual(exception.offendingEntry, entry)
        self.assertIsNone(exception.lineno)
        self.assertIsNone(exception.path)

    def test_addHostKey(self):
        """
        L{KnownHostsFile.addHostKey} adds a new L{HashedEntry} to the host
        file, and returns it.
        """
        hostsFile = self.loadSampleHostsFile()
        aKey = Key.fromString(thirdSampleKey)
        self.assertEqual(False, hostsFile.hasHostKey(b'somewhere.example.com', aKey))
        newEntry = hostsFile.addHostKey(b'somewhere.example.com', aKey)
        self.assertEqual(20, len(newEntry._hostSalt))
        self.assertEqual(True, newEntry.matchesHost(b'somewhere.example.com'))
        self.assertEqual(newEntry.keyType, b'ssh-rsa')
        self.assertEqual(aKey, newEntry.publicKey)
        self.assertEqual(True, hostsFile.hasHostKey(b'somewhere.example.com', aKey))

    def test_randomSalts(self):
        """
        L{KnownHostsFile.addHostKey} generates a random salt for each new key,
        so subsequent salts will be different.
        """
        hostsFile = self.loadSampleHostsFile()
        aKey = Key.fromString(thirdSampleKey)
        self.assertNotEqual(hostsFile.addHostKey(b'somewhere.example.com', aKey)._hostSalt, hostsFile.addHostKey(b'somewhere-else.example.com', aKey)._hostSalt)

    def test_verifyValidKey(self):
        """
        Verifying a valid key should return a L{Deferred} which fires with
        True.
        """
        hostsFile = self.loadSampleHostsFile()
        hostsFile.addHostKey(b'1.2.3.4', Key.fromString(sampleKey))
        ui = FakeUI()
        d = hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'1.2.3.4', Key.fromString(sampleKey))
        l = []
        d.addCallback(l.append)
        self.assertEqual(l, [True])

    def test_verifyInvalidKey(self):
        """
        Verifying an invalid key should return a L{Deferred} which fires with a
        L{HostKeyChanged} failure.
        """
        hostsFile = self.loadSampleHostsFile()
        wrongKey = Key.fromString(thirdSampleKey)
        ui = FakeUI()
        hostsFile.addHostKey(b'1.2.3.4', Key.fromString(sampleKey))
        d = hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'1.2.3.4', wrongKey)
        return self.assertFailure(d, HostKeyChanged)

    def verifyNonPresentKey(self):
        """
        Set up a test to verify a key that isn't present.  Return a 3-tuple of
        the UI, a list set up to collect the result of the verifyHostKey call,
        and the sample L{KnownHostsFile} being used.

        This utility method avoids returning a L{Deferred}, and records results
        in the returned list instead, because the events which get generated
        here are pre-recorded in the 'ui' object.  If the L{Deferred} in
        question does not fire, the it will fail quickly with an empty list.
        """
        hostsFile = self.loadSampleHostsFile()
        absentKey = Key.fromString(thirdSampleKey)
        ui = FakeUI()
        l = []
        d = hostsFile.verifyHostKey(ui, b'sample-host.example.com', b'4.3.2.1', absentKey)
        d.addBoth(l.append)
        self.assertEqual([], l)
        self.assertEqual(ui.promptText, b"The authenticity of host 'sample-host.example.com (4.3.2.1)' can't be established.\nRSA key fingerprint is SHA256:mS7mDBGhewdzJkaKRkx+wMjUdZb/GzvgcdoYjX5Js9I=.\nAre you sure you want to continue connecting (yes/no)? ")
        return (ui, l, hostsFile)

    def test_verifyNonPresentKey_Yes(self):
        """
        Verifying a key where neither the hostname nor the IP are present
        should result in the UI being prompted with a message explaining as
        much.  If the UI says yes, the Deferred should fire with True.
        """
        ui, l, knownHostsFile = self.verifyNonPresentKey()
        ui.promptDeferred.callback(True)
        self.assertEqual([True], l)
        reloaded = KnownHostsFile.fromPath(knownHostsFile.savePath)
        self.assertEqual(True, reloaded.hasHostKey(b'4.3.2.1', Key.fromString(thirdSampleKey)))
        self.assertEqual(True, reloaded.hasHostKey(b'sample-host.example.com', Key.fromString(thirdSampleKey)))

    def test_verifyNonPresentKey_No(self):
        """
        Verifying a key where neither the hostname nor the IP are present
        should result in the UI being prompted with a message explaining as
        much.  If the UI says no, the Deferred should fail with
        UserRejectedKey.
        """
        ui, l, knownHostsFile = self.verifyNonPresentKey()
        ui.promptDeferred.callback(False)
        l[0].trap(UserRejectedKey)

    def test_verifyNonPresentECKey(self):
        """
        Set up a test to verify an ECDSA key that isn't present.
        Return a 3-tuple of the UI, a list set up to collect the result
        of the verifyHostKey call, and the sample L{KnownHostsFile} being used.
        """
        ecObj = Key._fromECComponents(x=keydata.ECDatanistp256['x'], y=keydata.ECDatanistp256['y'], privateValue=keydata.ECDatanistp256['privateValue'], curve=keydata.ECDatanistp256['curve'])
        hostsFile = self.loadSampleHostsFile()
        ui = FakeUI()
        l = []
        d = hostsFile.verifyHostKey(ui, b'sample-host.example.com', b'4.3.2.1', ecObj)
        d.addBoth(l.append)
        self.assertEqual([], l)
        self.assertEqual(ui.promptText, b"The authenticity of host 'sample-host.example.com (4.3.2.1)' can't be established.\nECDSA key fingerprint is SHA256:fJnSpgCcYoYYsaBbnWj1YBghGh/QTDgfe4w4U5M5tEo=.\nAre you sure you want to continue connecting (yes/no)? ")

    def test_verifyHostIPMismatch(self):
        """
        Verifying a key where the host is present (and correct), but the IP is
        present and different, should result the deferred firing in a
        HostKeyChanged failure.
        """
        hostsFile = self.loadSampleHostsFile()
        wrongKey = Key.fromString(thirdSampleKey)
        ui = FakeUI()
        d = hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'4.3.2.1', wrongKey)
        return self.assertFailure(d, HostKeyChanged)

    def test_verifyKeyForHostAndIP(self):
        """
        Verifying a key where the hostname is present but the IP is not should
        result in the key being added for the IP and the user being warned
        about the change.
        """
        ui = FakeUI()
        hostsFile = self.loadSampleHostsFile()
        expectedKey = Key.fromString(sampleKey)
        hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'5.4.3.2', expectedKey)
        self.assertEqual(True, KnownHostsFile.fromPath(hostsFile.savePath).hasHostKey(b'5.4.3.2', expectedKey))
        self.assertEqual(["Warning: Permanently added the RSA host key for IP address '5.4.3.2' to the list of known hosts."], ui.userWarnings)

    def test_getHostKeyAlgorithms(self):
        """
        For a given host, get the host key algorithms for that
        host in the known_hosts file.
        """
        hostsFile = self.loadSampleHostsFile()
        hostsFile.addHostKey(b'www.twistedmatrix.com', Key.fromString(otherSampleKey))
        hostsFile.addHostKey(b'www.twistedmatrix.com', Key.fromString(ecdsaSampleKey))
        hostsFile.save()
        options = {}
        options['known-hosts'] = hostsFile.savePath.path
        algorithms = default.getHostKeyAlgorithms(b'www.twistedmatrix.com', options)
        expectedAlgorithms = [b'ssh-rsa', b'ecdsa-sha2-nistp256']
        self.assertEqual(algorithms, expectedAlgorithms)