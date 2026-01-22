from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.calliope import cli_tree_markdown as markdown
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.document_renderers import token_renderer
from prompt_toolkit.layout import controls
def GenerateHelpContent(cli, width):
    """Returns help lines for the current token."""
    if width > 80:
        width = 80
    doc = cli.current_buffer.document
    args = cli.parser.ParseCommand(doc.text_before_cursor)
    if not args:
        return []
    arg = args[-1]
    if arg.token_type in (parser.ArgTokenType.GROUP, parser.ArgTokenType.COMMAND):
        return GenerateHelpForCommand(cli, arg, width)
    elif arg.token_type == parser.ArgTokenType.FLAG:
        return GenerateHelpForFlag(cli, arg, width)
    elif arg.token_type == parser.ArgTokenType.FLAG_ARG:
        return GenerateHelpForFlag(cli, args[-2], width)
    elif arg.token_type == parser.ArgTokenType.POSITIONAL:
        return GenerateHelpForPositional(cli, arg, width)
    return []