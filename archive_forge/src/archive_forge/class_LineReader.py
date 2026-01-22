import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LineReader(object):
    """Reads a file line by line"""

    def __init__(self, filename):
        if isinstance(filename, file):
            self.file = filename
        else:
            self.file = codecs.open(filename, 'rU', 'utf-8')
        self.linenumber = 1
        self.lastline = None
        self.current = None
        self.mustread = True
        self.depleted = False
        try:
            self.readline()
        except UnicodeDecodeError:
            import gzip
            self.file = gzip.open(filename, 'rb')
            self.readline()

    def setstart(self, firstline):
        """Set the first line to read."""
        for i in range(firstline):
            self.file.readline()
        self.linenumber = firstline

    def setend(self, lastline):
        """Set the last line to read."""
        self.lastline = lastline

    def currentline(self):
        """Get the current line"""
        if self.mustread:
            self.readline()
        return self.current

    def nextline(self):
        """Go to next line"""
        if self.depleted:
            Trace.fatal('Read beyond file end')
        self.mustread = True

    def readline(self):
        """Read a line from elyxer.file"""
        self.current = self.file.readline()
        if not isinstance(self.file, codecs.StreamReaderWriter):
            self.current = self.current.decode('utf-8')
        if len(self.current) == 0:
            self.depleted = True
        self.current = self.current.rstrip('\n\r')
        self.linenumber += 1
        self.mustread = False
        Trace.prefix = 'Line ' + str(self.linenumber) + ': '
        if self.linenumber % 1000 == 0:
            Trace.message('Parsing')

    def finished(self):
        """Find out if the file is finished"""
        if self.lastline and self.linenumber == self.lastline:
            return True
        if self.mustread:
            self.readline()
        return self.depleted

    def close(self):
        self.file.close()