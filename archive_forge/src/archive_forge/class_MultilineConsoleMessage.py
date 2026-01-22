from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
class MultilineConsoleMessage(object):
    """A multiline implementation of ConsoleMessage."""

    def __init__(self, message, stream, indentation_level=0):
        """Constructor.

    Args:
      message: str, the message that this object represents.
      stream: The output stream to write to.
      indentation_level: int, The indentation level of the message. Each
        indentation is represented by two spaces.
    """
        self._stream = stream
        self._console_attr = console_attr.GetConsoleAttr()
        self._console_width = self._console_attr.GetTermSize()[0] - 1
        if self._console_width < 0:
            self._console_width = 0
        self._level = indentation_level
        self._no_output = False
        if self._console_width - INDENTATION_WIDTH * indentation_level <= 0:
            self._no_output = True
        self._message = None
        self._lines = []
        self._has_update = False
        self._num_lines_changed = False
        self._UpdateMessage(message)

    @property
    def lines(self):
        return self._lines

    @property
    def num_lines(self):
        return len(self._lines)

    @property
    def has_update(self):
        return self._has_update

    @property
    def num_lines_changed(self):
        return self._num_lines_changed

    def _UpdateMessage(self, new_message):
        """Updates the message for this Message object."""
        if not isinstance(new_message, six.string_types):
            raise TypeError('expected a string or other character buffer object')
        if new_message != self._message:
            self._message = new_message
            if self._no_output:
                return
            num_old_lines = len(self._lines)
            self._lines = self._SplitMessageIntoLines(self._message)
            self._has_update = True
            self._num_lines_changed = num_old_lines != len(self._lines)

    def _SplitMessageIntoLines(self, message):
        """Converts message into a list of strs, each representing a line."""
        lines = self._console_attr.SplitLine(message, self.effective_width)
        for i in range(len(lines)):
            lines[i] += '\n'
        return lines

    def Print(self):
        """Prints out the message to the console.

    The implementation of this function assumes that when called, the
    cursor position of the terminal is where the message should start printing.
    """
        if self._no_output:
            return
        for line in self._lines:
            self._ClearLine()
            self._WriteLine(line)
        self._has_update = False

    @property
    def effective_width(self):
        """The effective width when the indentation level is considered."""
        return self._console_width - INDENTATION_WIDTH * self._level

    def _ClearLine(self):
        self._stream.write('\r{}\r'.format(' ' * self._console_width))

    def _WriteLine(self, line):
        self._stream.write(self._level * INDENTATION_WIDTH * ' ' + line)