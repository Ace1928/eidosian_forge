import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class FTPFileListingTests(TestCase):

    def getFilesForLines(self, lines):
        fileList = MyFTPFileListProtocol()
        d = loopback.loopbackAsync(PrintLines(lines), fileList)
        d.addCallback(lambda _: (fileList.files, fileList.other))
        return d

    def test_OneLine(self):
        """
        This example line taken from the docstring for FTPFileListProtocol

        @return: L{Deferred} of command response
        """
        line = '-rw-r--r--   1 root     other        531 Jan 29 03:26 README'

        def check(fileOther):
            (file,), other = fileOther
            self.assertFalse(other, f'unexpect unparsable lines: {repr(other)}')
            self.assertTrue(file['filetype'] == '-', 'misparsed fileitem')
            self.assertTrue(file['perms'] == 'rw-r--r--', 'misparsed perms')
            self.assertTrue(file['owner'] == 'root', 'misparsed fileitem')
            self.assertTrue(file['group'] == 'other', 'misparsed fileitem')
            self.assertTrue(file['size'] == 531, 'misparsed fileitem')
            self.assertTrue(file['date'] == 'Jan 29 03:26', 'misparsed fileitem')
            self.assertTrue(file['filename'] == 'README', 'misparsed fileitem')
            self.assertTrue(file['nlinks'] == 1, 'misparsed nlinks')
            self.assertFalse(file['linktarget'], 'misparsed linktarget')
        return self.getFilesForLines([line]).addCallback(check)

    def test_VariantLines(self):
        """
        Variant lines.
        """
        line1 = 'drw-r--r--   2 root     other        531 Jan  9  2003 A'
        line2 = 'lrw-r--r--   1 root     other          1 Jan 29 03:26 B -> A'
        line3 = 'woohoo! '

        def check(result):
            (file1, file2), (other,) = result
            self.assertTrue(other == 'woohoo! \r', 'incorrect other line')
            self.assertTrue(file1['filetype'] == 'd', 'misparsed fileitem')
            self.assertTrue(file1['perms'] == 'rw-r--r--', 'misparsed perms')
            self.assertTrue(file1['owner'] == 'root', 'misparsed owner')
            self.assertTrue(file1['group'] == 'other', 'misparsed group')
            self.assertTrue(file1['size'] == 531, 'misparsed size')
            self.assertTrue(file1['date'] == 'Jan  9  2003', 'misparsed date')
            self.assertTrue(file1['filename'] == 'A', 'misparsed filename')
            self.assertTrue(file1['nlinks'] == 2, 'misparsed nlinks')
            self.assertFalse(file1['linktarget'], 'misparsed linktarget')
            self.assertTrue(file2['filetype'] == 'l', 'misparsed fileitem')
            self.assertTrue(file2['perms'] == 'rw-r--r--', 'misparsed perms')
            self.assertTrue(file2['owner'] == 'root', 'misparsed owner')
            self.assertTrue(file2['group'] == 'other', 'misparsed group')
            self.assertTrue(file2['size'] == 1, 'misparsed size')
            self.assertTrue(file2['date'] == 'Jan 29 03:26', 'misparsed date')
            self.assertTrue(file2['filename'] == 'B', 'misparsed filename')
            self.assertTrue(file2['nlinks'] == 1, 'misparsed nlinks')
            self.assertTrue(file2['linktarget'] == 'A', 'misparsed linktarget')
        return self.getFilesForLines([line1, line2, line3]).addCallback(check)

    def test_UnknownLine(self):
        """
        Unknown lines.
        """

        def check(result):
            files, others = result
            self.assertFalse(files, 'unexpected file entries')
            self.assertTrue(others == ['ABC\r', 'not a file\r'], 'incorrect unparsable lines: %s' % repr(others))
        return self.getFilesForLines(['ABC', 'not a file']).addCallback(check)

    def test_filenameWithUnescapedSpace(self):
        """
        Will parse filenames and linktargets containing unescaped
        space characters.
        """
        line1 = 'drw-r--r--   2 root     other        531 Jan  9  2003 A B'
        line2 = 'lrw-r--r--   1 root     other          1 Jan 29 03:26 B A -> D C/A B'

        def check(result):
            files, others = result
            self.assertEqual([], others, 'unexpected others entries')
            self.assertEqual('A B', files[0]['filename'], 'misparsed filename')
            self.assertEqual('B A', files[1]['filename'], 'misparsed filename')
            self.assertEqual('D C/A B', files[1]['linktarget'], 'misparsed linktarget')
        return self.getFilesForLines([line1, line2]).addCallback(check)

    def test_filenameWithEscapedSpace(self):
        """
        Will parse filenames and linktargets containing escaped
        space characters.
        """
        line1 = 'drw-r--r--   2 root     other        531 Jan  9  2003 A\\ B'
        line2 = 'lrw-r--r--   1 root     other          1 Jan 29 03:26 B A -> D\\ C/A B'

        def check(result):
            files, others = result
            self.assertEqual([], others, 'unexpected others entries')
            self.assertEqual('A B', files[0]['filename'], 'misparsed filename')
            self.assertEqual('B A', files[1]['filename'], 'misparsed filename')
            self.assertEqual('D C/A B', files[1]['linktarget'], 'misparsed linktarget')
        return self.getFilesForLines([line1, line2]).addCallback(check)

    def test_Year(self):
        """
        This example derived from bug description in issue 514.

        @return: L{Deferred} of command response
        """
        fileList = ftp.FTPFileListProtocol()
        exampleLine = b'-rw-r--r--   1 root     other        531 Jan 29 2003 README\n'

        class PrintLine(protocol.Protocol):

            def connectionMade(self):
                self.transport.write(exampleLine)
                self.transport.loseConnection()

        def check(ignored):
            file = fileList.files[0]
            self.assertTrue(file['size'] == 531, 'misparsed fileitem')
            self.assertTrue(file['date'] == 'Jan 29 2003', 'misparsed fileitem')
            self.assertTrue(file['filename'] == 'README', 'misparsed fileitem')
        d = loopback.loopbackAsync(PrintLine(), fileList)
        return d.addCallback(check)