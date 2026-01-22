import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
@implementer(interfaces.IConsumer)
class DTP(protocol.Protocol):
    isConnected = False
    _cons = None
    _onConnLost = None
    _buffer = None
    _encoding = 'latin-1'

    def connectionMade(self):
        self.isConnected = True
        self.factory.deferred.callback(None)
        self._buffer = []

    def connectionLost(self, reason):
        self.isConnected = False
        if self._onConnLost is not None:
            self._onConnLost.callback(None)

    def sendLine(self, line):
        """
        Send a line to data channel.

        @param line: The line to be sent.
        @type line: L{bytes}
        """
        self.transport.write(line + b'\r\n')

    def _formatOneListResponse(self, name, size, directory, permissions, hardlinks, modified, owner, group):
        """
        Helper method to format one entry's info into a text entry like:
        'drwxrwxrwx   0 user   group   0 Jan 01  1970 filename.txt'

        @param name: C{bytes} name of the entry (file or directory or link)
        @param size: C{int} size of the entry
        @param directory: evals to C{bool} - whether the entry is a directory
        @param permissions: L{twisted.python.filepath.Permissions} object
            representing that entry's permissions
        @param hardlinks: C{int} number of hardlinks
        @param modified: C{float} - entry's last modified time in seconds
            since the epoch
        @param owner: C{str} username of the owner
        @param group: C{str} group name of the owner

        @return: C{str} in the requisite format
        """

        def formatDate(mtime):
            now = time.gmtime()
            info = {'month': _months[mtime.tm_mon], 'day': mtime.tm_mday, 'year': mtime.tm_year, 'hour': mtime.tm_hour, 'minute': mtime.tm_min}
            if now.tm_year != mtime.tm_year:
                return '%(month)s %(day)02d %(year)5d' % info
            else:
                return '%(month)s %(day)02d %(hour)02d:%(minute)02d' % info
        format = '%(directory)s%(permissions)s%(hardlinks)4d %(owner)-9s %(group)-9s %(size)15d %(date)12s '
        msg = (format % {'directory': directory and 'd' or '-', 'permissions': permissions.shorthand(), 'hardlinks': hardlinks, 'owner': owner[:8], 'group': group[:8], 'size': size, 'date': formatDate(time.gmtime(modified))}).encode(self._encoding)
        return msg + name

    def sendListResponse(self, name, response):
        self.sendLine(self._formatOneListResponse(name, *response))

    def registerProducer(self, producer, streaming):
        return self.transport.registerProducer(producer, streaming)

    def unregisterProducer(self):
        self.transport.unregisterProducer()
        self.transport.loseConnection()

    def write(self, data):
        if self.isConnected:
            return self.transport.write(data)
        raise Exception('Crap damn crap damn crap damn')

    def _conswrite(self, bytes):
        try:
            self._cons.write(bytes)
        except BaseException:
            self._onConnLost.errback()

    def dataReceived(self, bytes):
        if self._cons is not None:
            self._conswrite(bytes)
        else:
            self._buffer.append(bytes)

    def _unregConsumer(self, ignored):
        self._cons.unregisterProducer()
        self._cons = None
        del self._onConnLost
        return ignored

    def registerConsumer(self, cons):
        assert self._cons is None
        self._cons = cons
        self._cons.registerProducer(self, True)
        for chunk in self._buffer:
            self._conswrite(chunk)
        self._buffer = None
        if self.isConnected:
            self._onConnLost = d = defer.Deferred()
            d.addBoth(self._unregConsumer)
            return d
        else:
            self._cons.unregisterProducer()
            self._cons = None
            return defer.succeed(None)

    def resumeProducing(self):
        self.transport.resumeProducing()

    def pauseProducing(self):
        self.transport.pauseProducing()

    def stopProducing(self):
        self.transport.stopProducing()