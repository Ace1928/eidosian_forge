import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class BracketProcessor(MathsProcessor):
    """A processor for bracket commands."""

    def process(self, contents, index):
        """Convert the bracket using Unicode pieces, if possible."""
        if Options.simplemath:
            return
        if self.checkleft(contents, index):
            return self.processleft(contents, index)

    def processleft(self, contents, index):
        """Process a left bracket."""
        rightindex = self.findright(contents, index + 1)
        if not rightindex:
            return
        size = self.findmax(contents, index, rightindex)
        self.resize(contents[index], size)
        self.resize(contents[rightindex], size)

    def checkleft(self, contents, index):
        """Check if the command at the given index is left."""
        return self.checkdirection(contents[index], '\\left')

    def checkright(self, contents, index):
        """Check if the command at the given index is right."""
        return self.checkdirection(contents[index], '\\right')

    def checkdirection(self, bit, command):
        """Check if the given bit is the desired bracket command."""
        if not isinstance(bit, BracketCommand):
            return False
        return bit.command == command

    def findright(self, contents, index):
        """Find the right bracket starting at the given index, or 0."""
        depth = 1
        while index < len(contents):
            if self.checkleft(contents, index):
                depth += 1
            if self.checkright(contents, index):
                depth -= 1
            if depth == 0:
                return index
            index += 1
        return None

    def findmax(self, contents, leftindex, rightindex):
        """Find the max size of the contents between the two given indices."""
        sliced = contents[leftindex:rightindex]
        return max([element.size for element in sliced])

    def resize(self, command, size):
        """Resize a bracket command to the given size."""
        character = command.extracttext()
        alignment = command.command.replace('\\', '')
        bracket = BigBracket(size, character, alignment)
        command.output = ContentsOutput()
        command.contents = bracket.getcontents()