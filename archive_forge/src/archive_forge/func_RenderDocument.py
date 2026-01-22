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
def RenderDocument(style='text', fin=None, out=None, width=80, notes=None, title=None, command_metadata=None, command_node=None):
    """Renders markdown to a selected document style.

  Args:
    style: The rendered document style name, must be one of the STYLES keys.
    fin: The input stream containing the markdown.
    out: The output stream for the rendered document.
    width: The page width in characters.
    notes: Optional sentences inserted in the NOTES section.
    title: The document title.
    command_metadata: Optional metadata of command, including available flags.
    command_node: The command object that the document is being rendered for.

  Raises:
    DocumentStyleError: The markdown style was unknown.
  """
    if style not in STYLES:
        raise DocumentStyleError(style)
    style_renderer = STYLES[style](out=out or sys.stdout, title=title, width=width, command_metadata=command_metadata, command_node=command_node)
    MarkdownRenderer(style_renderer, fin=fin or sys.stdin, notes=notes, command_metadata=command_metadata).Run()