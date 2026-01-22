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
class FTPFileListProtocol(basic.LineReceiver):
    """
    Parser for standard FTP file listings

    This is the evil required to match::

        -rw-r--r--   1 root     other        531 Jan 29 03:26 README

    If you need different evil for a wacky FTP server, you can
    override either C{fileLinePattern} or C{parseDirectoryLine()}.

    It populates the instance attribute self.files, which is a list containing
    dicts with the following keys (examples from the above line):
        - filetype:   e.g. 'd' for directories, or '-' for an ordinary file
        - perms:      e.g. 'rw-r--r--'
        - nlinks:     e.g. 1
        - owner:      e.g. 'root'
        - group:      e.g. 'other'
        - size:       e.g. 531
        - date:       e.g. 'Jan 29 03:26'
        - filename:   e.g. 'README'
        - linktarget: e.g. 'some/file'

    Note that the 'date' value will be formatted differently depending on the
    date.  Check U{http://cr.yp.to/ftp.html} if you really want to try to parse
    it.

    It also matches the following::
        -rw-r--r--   1 root     other        531 Jan 29 03:26 I HAVE\\ SPACE
           - filename:   e.g. 'I HAVE SPACE'

        -rw-r--r--   1 root     other        531 Jan 29 03:26 LINK -> TARGET
           - filename:   e.g. 'LINK'
           - linktarget: e.g. 'TARGET'

        -rw-r--r--   1 root     other        531 Jan 29 03:26 N S -> L S
           - filename:   e.g. 'N S'
           - linktarget: e.g. 'L S'

    @ivar files: list of dicts describing the files in this listing
    """
    fileLinePattern = re.compile('^(?P<filetype>.)(?P<perms>.{9})\\s+(?P<nlinks>\\d*)\\s*(?P<owner>\\S+)\\s+(?P<group>\\S+)\\s+(?P<size>\\d+)\\s+(?P<date>...\\s+\\d+\\s+[\\d:]+)\\s+(?P<filename>.{1,}?)( -> (?P<linktarget>[^\\r]*))?\\r?$')
    delimiter = b'\n'
    _encoding = 'latin-1'

    def __init__(self):
        self.files = []

    def lineReceived(self, line):
        if bytes != str:
            line = line.decode(self._encoding)
        d = self.parseDirectoryLine(line)
        if d is None:
            self.unknownLine(line)
        else:
            self.addFile(d)

    def parseDirectoryLine(self, line):
        """
        Return a dictionary of fields, or None if line cannot be parsed.

        @param line: line of text expected to contain a directory entry
        @type line: str

        @return: dict
        """
        match = self.fileLinePattern.match(line)
        if match is None:
            return None
        else:
            d = match.groupdict()
            d['filename'] = d['filename'].replace('\\ ', ' ')
            d['nlinks'] = int(d['nlinks'])
            d['size'] = int(d['size'])
            if d['linktarget']:
                d['linktarget'] = d['linktarget'].replace('\\ ', ' ')
            return d

    def addFile(self, info):
        """
        Append file information dictionary to the list of known files.

        Subclasses can override or extend this method to handle file
        information differently without affecting the parsing of data
        from the server.

        @param info: dictionary containing the parsed representation
                     of the file information
        @type info: dict
        """
        self.files.append(info)

    def unknownLine(self, line):
        """
        Deal with received lines which could not be parsed as file
        information.

        Subclasses can override this to perform any special processing
        needed.

        @param line: unparsable line as received
        @type line: str
        """
        pass