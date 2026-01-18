from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def text_wrap(text, length=None, indent='', firstline_indent=None):
    """Wraps a given text to a maximum line length and returns it.

  It turns lines that only contain whitespace into empty lines, keeps new lines,
  and expands tabs using 4 spaces.

  Args:
    text: str, text to wrap.
    length: int, maximum length of a line, includes indentation.
        If this is None then use get_help_width()
    indent: str, indent for all but first line.
    firstline_indent: str, indent for first line; if None, fall back to indent.

  Returns:
    str, the wrapped text.

  Raises:
    ValueError: Raised if indent or firstline_indent not shorter than length.
  """
    if length is None:
        length = get_help_width()
    if indent is None:
        indent = ''
    if firstline_indent is None:
        firstline_indent = indent
    if len(indent) >= length:
        raise ValueError('Length of indent exceeds length')
    if len(firstline_indent) >= length:
        raise ValueError('Length of first line indent exceeds length')
    text = text.expandtabs(4)
    result = []
    wrapper = textwrap.TextWrapper(width=length, initial_indent=firstline_indent, subsequent_indent=indent)
    subsequent_wrapper = textwrap.TextWrapper(width=length, initial_indent=indent, subsequent_indent=indent)
    for paragraph in (p.strip() for p in text.splitlines()):
        if paragraph:
            result.extend(wrapper.wrap(paragraph))
        else:
            result.append('')
        wrapper = subsequent_wrapper
    return '\n'.join(result)