import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
class StateMachineWS(StateMachine):
    """
    `StateMachine` subclass specialized for whitespace recognition.

    There are three methods provided for extracting indented text blocks:

    - `get_indented()`: use when the indent is unknown.
    - `get_known_indented()`: use when the indent is known for all lines.
    - `get_first_known_indented()`: use when only the first line's indent is
      known.
    """

    def get_indented(self, until_blank=False, strip_indent=True):
        """
        Return a block of indented lines of text, and info.

        Extract an indented block where the indent is unknown for all lines.

        :Parameters:
            - `until_blank`: Stop collecting at the first blank line if true.
            - `strip_indent`: Strip common leading indent if true (default).

        :Return:
            - the indented block (a list of lines of text),
            - its indent,
            - its first line offset from BOF, and
            - whether or not it finished with a blank line.
        """
        offset = self.abs_line_offset()
        indented, indent, blank_finish = self.input_lines.get_indented(self.line_offset, until_blank, strip_indent)
        if indented:
            self.next_line(len(indented) - 1)
        while indented and (not indented[0].strip()):
            indented.trim_start()
            offset += 1
        return (indented, indent, offset, blank_finish)

    def get_known_indented(self, indent, until_blank=False, strip_indent=True):
        """
        Return an indented block and info.

        Extract an indented block where the indent is known for all lines.
        Starting with the current line, extract the entire text block with at
        least `indent` indentation (which must be whitespace, except for the
        first line).

        :Parameters:
            - `indent`: The number of indent columns/characters.
            - `until_blank`: Stop collecting at the first blank line if true.
            - `strip_indent`: Strip `indent` characters of indentation if true
              (default).

        :Return:
            - the indented block,
            - its first line offset from BOF, and
            - whether or not it finished with a blank line.
        """
        offset = self.abs_line_offset()
        indented, indent, blank_finish = self.input_lines.get_indented(self.line_offset, until_blank, strip_indent, block_indent=indent)
        self.next_line(len(indented) - 1)
        while indented and (not indented[0].strip()):
            indented.trim_start()
            offset += 1
        return (indented, offset, blank_finish)

    def get_first_known_indented(self, indent, until_blank=False, strip_indent=True, strip_top=True):
        """
        Return an indented block and info.

        Extract an indented block where the indent is known for the first line
        and unknown for all other lines.

        :Parameters:
            - `indent`: The first line's indent (# of columns/characters).
            - `until_blank`: Stop collecting at the first blank line if true
              (1).
            - `strip_indent`: Strip `indent` characters of indentation if true
              (1, default).
            - `strip_top`: Strip blank lines from the beginning of the block.

        :Return:
            - the indented block,
            - its indent,
            - its first line offset from BOF, and
            - whether or not it finished with a blank line.
        """
        offset = self.abs_line_offset()
        indented, indent, blank_finish = self.input_lines.get_indented(self.line_offset, until_blank, strip_indent, first_indent=indent)
        self.next_line(len(indented) - 1)
        if strip_top:
            while indented and (not indented[0].strip()):
                indented.trim_start()
                offset += 1
        return (indented, indent, offset, blank_finish)