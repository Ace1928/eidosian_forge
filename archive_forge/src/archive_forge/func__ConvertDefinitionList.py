from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import re
import sys
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.document_renderers import devsite_renderer
from googlecloudsdk.core.document_renderers import html_renderer
from googlecloudsdk.core.document_renderers import linter_renderer
from googlecloudsdk.core.document_renderers import man_renderer
from googlecloudsdk.core.document_renderers import markdown_renderer
from googlecloudsdk.core.document_renderers import renderer
from googlecloudsdk.core.document_renderers import text_renderer
def _ConvertDefinitionList(self, i):
    """Detects and converts a definition list item markdown line.

         [item-level-1]:: [definition-line]
         [definition-lines]
         [item-level-2]::: [definition-line]
         [definition-lines]

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input line is a definition list item markdown, i otherwise.
    """
    if i:
        return i
    index_at_definition_markdown = self._line.find('::')
    if index_at_definition_markdown < 0:
        return i
    level = 1
    list_level = None
    original_i = i
    i = index_at_definition_markdown + 2
    while i < len(self._line) and self._line[i] == ':':
        i += 1
        level += 1
    if i < len(self._line) and (not self._line[i].isspace()):
        return original_i
    while i < len(self._line) and self._line[i].isspace():
        i += 1
    end = i >= len(self._line) and (not index_at_definition_markdown)
    if end:
        level -= 1
    if self._line.endswith('::'):
        self._last_list_level = level
    elif self._last_list_level and (not self._line.startswith('::')):
        list_level = self._last_list_level + 1
    if self._lists[self._depth].bullet or self._lists[self._depth].level < level:
        self._depth += 1
        if self._depth >= len(self._lists):
            self._lists.append(_ListElementState())
    else:
        while self._lists[self._depth].level > level:
            self._depth -= 1
    self._Fill()
    if end:
        i = len(self._line)
        definition = None
    else:
        self._lists[self._depth].bullet = False
        self._lists[self._depth].ignore_line = 2
        self._lists[self._depth].level = level
        self._buf = self._line[:index_at_definition_markdown]
        definition = self._Attributes()
    if list_level:
        level = list_level
    self._renderer.List(level, definition=definition, end=end)
    if i < len(self._line):
        self._buf += self._line[i:]
    return -1