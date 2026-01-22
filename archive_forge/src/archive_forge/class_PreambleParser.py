import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class PreambleParser(Parser):
    """A parser for the LyX preamble."""
    preamble = []

    def parse(self, reader):
        """Parse the full preamble with all statements."""
        self.ending = HeaderConfig.parameters['endpreamble']
        self.parseending(reader, lambda: self.parsepreambleline(reader))
        return []

    def parsepreambleline(self, reader):
        """Parse a single preamble line."""
        PreambleParser.preamble.append(reader.currentline())
        reader.nextline()