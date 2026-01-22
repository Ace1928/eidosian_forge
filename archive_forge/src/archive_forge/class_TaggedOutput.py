import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class TaggedOutput(ContentsOutput):
    """Outputs an HTML tag surrounding the contents."""
    tag = None
    breaklines = False
    empty = False

    def settag(self, tag, breaklines=False, empty=False):
        """Set the value for the tag and other attributes."""
        self.tag = tag
        if breaklines:
            self.breaklines = breaklines
        if empty:
            self.empty = empty
        return self

    def setbreaklines(self, breaklines):
        """Set the value for breaklines."""
        self.breaklines = breaklines
        return self

    def gethtml(self, container):
        """Return the HTML code."""
        if self.empty:
            return [self.selfclosing(container)]
        html = [self.open(container)]
        html += ContentsOutput.gethtml(self, container)
        html.append(self.close(container))
        return html

    def open(self, container):
        """Get opening line."""
        if not self.checktag():
            return ''
        open = '<' + self.tag + '>'
        if self.breaklines:
            return open + '\n'
        return open

    def close(self, container):
        """Get closing line."""
        if not self.checktag():
            return ''
        close = '</' + self.tag.split()[0] + '>'
        if self.breaklines:
            return '\n' + close + '\n'
        return close

    def selfclosing(self, container):
        """Get self-closing line."""
        if not self.checktag():
            return ''
        selfclosing = '<' + self.tag + '/>'
        if self.breaklines:
            return selfclosing + '\n'
        return selfclosing

    def checktag(self):
        """Check that the tag is valid."""
        if not self.tag:
            Trace.error('No tag in ' + str(container))
            return False
        if self.tag == '':
            return False
        return True