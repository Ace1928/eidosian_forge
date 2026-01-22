from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def IndentAsciiDoc(text, depth=0):
    """Tabs over all lines in text using ascii doc syntax."""
    additional_tabs = ':' * depth
    return text.replace(ASCII_INDENT, additional_tabs + ASCII_INDENT)