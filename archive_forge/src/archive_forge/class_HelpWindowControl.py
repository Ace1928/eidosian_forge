from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.calliope import cli_tree_markdown as markdown
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.document_renderers import token_renderer
from prompt_toolkit.layout import controls
class HelpWindowControl(controls.UIControl):
    """Implementation of the help window."""

    def __init__(self, default_char=None):
        self._default_char = default_char

    def create_content(self, cli, width, height):
        data = GenerateHelpContent(cli, width)
        return controls.UIContent(lambda i: data[i], line_count=len(data), show_cursor=False, default_char=self._default_char)