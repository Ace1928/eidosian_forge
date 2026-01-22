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
class FTPServerPasvDataConnectionTests(FTPServerTestCase):
    """
    PASV data connection.
    """

    def _makeDataConnection(self, ignored=None):
        """
        Establish a passive data connection (i.e. client connecting to
        server).

        @param ignored: ignored
        @return: L{Deferred.addCallback}
        """
        d = self.client.queueStringCommand('PASV')

        def gotPASV(responseLines):
            host, port = ftp.decodeHostPort(responseLines[-1][4:])
            cc = protocol.ClientCreator(reactor, _BufferingProtocol)
            return cc.connectTCP('127.0.0.1', port)
        return d.addCallback(gotPASV)

    def _download(self, command, chainDeferred=None):
        """
        Download file.

        @param command: command to run
        @param chainDeferred: L{Deferred} used to queue commands.
        @return: L{Deferred} of command response
        """
        if chainDeferred is None:
            chainDeferred = defer.succeed(None)
        chainDeferred.addCallback(self._makeDataConnection)

        def queueCommand(downloader):
            d1 = self.client.queueStringCommand(command)
            d2 = downloader.d
            return defer.gatherResults([d1, d2])
        chainDeferred.addCallback(queueCommand)

        def downloadDone(result):
            ignored, downloader = result
            return downloader.buffer
        return chainDeferred.addCallback(downloadDone)

    def test_LISTEmpty(self):
        """
        When listing empty folders, LIST returns an empty response.
        """
        d = self._anonymousLogin()
        self._download('LIST', chainDeferred=d)

        def checkEmpty(result):
            self.assertEqual(b'', result)
        return d.addCallback(checkEmpty)

    def test_LISTWithBinLsFlags(self):
        """
        LIST ignores requests for folder with names like '-al' and will list
        the content of current folder.
        """
        os.mkdir(os.path.join(self.directory, 'foo'))
        os.mkdir(os.path.join(self.directory, 'bar'))
        d = self._anonymousLogin()
        self._download('LIST -aL', chainDeferred=d)

        def checkDownload(download):
            names = []
            for line in download.splitlines():
                names.append(line.split(b' ')[-1])
            self.assertEqual(2, len(names))
            self.assertIn(b'foo', names)
            self.assertIn(b'bar', names)
        return d.addCallback(checkDownload)

    def test_LISTWithContent(self):
        """
        LIST returns all folder's members, each member listed on a separate
        line and with name and other details.
        """
        os.mkdir(os.path.join(self.directory, 'foo'))
        os.mkdir(os.path.join(self.directory, 'bar'))
        d = self._anonymousLogin()
        self._download('LIST', chainDeferred=d)

        def checkDownload(download):
            self.assertEqual(2, len(download[:-2].split(b'\r\n')))
        d.addCallback(checkDownload)
        self._download('NLST ', chainDeferred=d)

        def checkDownload(download):
            filenames = download[:-2].split(b'\r\n')
            filenames.sort()
            self.assertEqual([b'bar', b'foo'], filenames)
        d.addCallback(checkDownload)
        self._download('LIST foo', chainDeferred=d)

        def checkDownload(download):
            self.assertEqual(b'', download)
        d.addCallback(checkDownload)

        def chdir(ignored):
            return self.client.queueStringCommand('CWD foo')
        d.addCallback(chdir)
        self._download('LIST', chainDeferred=d)

        def checkDownload(download):
            self.assertEqual(b'', download)
        return d.addCallback(checkDownload)

    def _listTestHelper(self, command, listOutput, expectedOutput):
        """
        Exercise handling by the implementation of I{LIST} or I{NLST} of certain
        return values and types from an L{IFTPShell.list} implementation.

        This will issue C{command} and assert that if the L{IFTPShell.list}
        implementation includes C{listOutput} as one of the file entries then
        the result given to the client is matches C{expectedOutput}.

        @param command: Either C{b"LIST"} or C{b"NLST"}
        @type command: L{bytes}

        @param listOutput: A value suitable to be used as an element of the list
            returned by L{IFTPShell.list}.  Vary the values and types of the
            contents to exercise different code paths in the server's handling
            of this result.

        @param expectedOutput: A line of output to expect as a result of
            C{listOutput} being transformed into a response to the command
            issued.
        @type expectedOutput: L{bytes}

        @return: A L{Deferred} which fires when the test is done, either with an
            L{Failure} if the test failed or with a function object if it
            succeeds.  The function object is the function which implements
            L{IFTPShell.list} (and is useful to make assertions about what
            warnings might have been emitted).
        @rtype: L{Deferred}
        """
        d = self._anonymousLogin()

        def patchedList(segments, keys=()):
            return defer.succeed([listOutput])

        def loggedIn(result):
            self.serverProtocol.shell.list = patchedList
            return result
        d.addCallback(loggedIn)
        self._download(f'{command} something', chainDeferred=d)

        def checkDownload(download):
            self.assertEqual(expectedOutput, download)
            return patchedList
        return d.addCallback(checkDownload)

    def test_LISTUnicode(self):
        """
        Unicode filenames returned from L{IFTPShell.list} are encoded using
        UTF-8 before being sent with the response.
        """
        return self._listTestHelper('LIST', ('my resumé', (0, 1, filepath.Permissions(511), 0, 0, 'user', 'group')), b'drwxrwxrwx   0 user      group                   0 Jan 01  1970 my resum\xc3\xa9\r\n')

    def test_LISTNonASCIIBytes(self):
        """
        When LIST receive a filename as byte string from L{IFTPShell.list}
        it will just pass the data to lower level without any change.

        @return: L{_listTestHelper}
        """
        return self._listTestHelper('LIST', (b'my resum\xc3\xa9', (0, 1, filepath.Permissions(511), 0, 0, 'user', 'group')), b'drwxrwxrwx   0 user      group                   0 Jan 01  1970 my resum\xc3\xa9\r\n')

    def test_ManyLargeDownloads(self):
        """
        Download many large files.

        @return: L{Deferred}
        """
        d = self._anonymousLogin()
        for size in range(100000, 110000, 500):
            with open(os.path.join(self.directory, '%d.txt' % (size,)), 'wb') as fObj:
                fObj.write(b'x' * size)
            self._download('RETR %d.txt' % (size,), chainDeferred=d)

            def checkDownload(download, size=size):
                self.assertEqual(size, len(download))
            d.addCallback(checkDownload)
        return d

    def test_downloadFolder(self):
        """
        When RETR is called for a folder, it will fail complaining that
        the path is a folder.
        """
        self.dirPath.child('foo').createDirectory()
        d = self._anonymousLogin()
        d.addCallback(self._makeDataConnection)

        def retrFolder(downloader):
            downloader.transport.loseConnection()
            deferred = self.client.queueStringCommand('RETR foo')
            return deferred
        d.addCallback(retrFolder)

        def failOnSuccess(result):
            raise AssertionError('Downloading a folder should not succeed.')
        d.addCallback(failOnSuccess)

        def checkError(failure):
            failure.trap(ftp.CommandFailed)
            self.assertEqual(['550 foo: is a directory'], failure.value.args[0])
            current_errors = self.flushLoggedErrors()
            self.assertEqual(0, len(current_errors), 'No errors should be logged while downloading a folder.')
        d.addErrback(checkError)
        return d

    def test_NLSTEmpty(self):
        """
        NLST with no argument returns the directory listing for the current
        working directory.
        """
        d = self._anonymousLogin()
        self.dirPath.child('test.txt').touch()
        self.dirPath.child('foo').createDirectory()
        self._download('NLST ', chainDeferred=d)

        def checkDownload(download):
            filenames = download[:-2].split(b'\r\n')
            filenames.sort()
            self.assertEqual([b'foo', b'test.txt'], filenames)
        return d.addCallback(checkDownload)

    def test_NLSTNonexistent(self):
        """
        NLST on a non-existent file/directory returns nothing.
        """
        d = self._anonymousLogin()
        self._download('NLST nonexistent.txt', chainDeferred=d)

        def checkDownload(download):
            self.assertEqual(b'', download)
        return d.addCallback(checkDownload)

    def test_NLSTUnicode(self):
        """
        NLST will receive Unicode filenames for IFTPShell.list, and will
        encode them using UTF-8.
        """
        return self._listTestHelper('NLST', ('my resumé', (0, 1, filepath.Permissions(511), 0, 0, 'user', 'group')), b'my resum\xc3\xa9\r\n')

    def test_NLSTNonASCIIBytes(self):
        """
        NLST will just pass the non-Unicode data to lower level.
        """
        return self._listTestHelper('NLST', (b'my resum\xc3\xa9', (0, 1, filepath.Permissions(511), 0, 0, 'user', 'group')), b'my resum\xc3\xa9\r\n')

    def test_NLSTOnPathToFile(self):
        """
        NLST on an existent file returns only the path to that file.
        """
        d = self._anonymousLogin()
        self.dirPath.child('test.txt').touch()
        self._download('NLST test.txt', chainDeferred=d)

        def checkDownload(download):
            filenames = download[:-2].split(b'\r\n')
            self.assertEqual([b'test.txt'], filenames)
        return d.addCallback(checkDownload)