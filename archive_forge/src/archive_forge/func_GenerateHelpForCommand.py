from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.calliope import cli_tree_markdown as markdown
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.document_renderers import token_renderer
from prompt_toolkit.layout import controls
def GenerateHelpForCommand(cli, token, width):
    """Returns help lines for a command token."""
    lines = []
    height = 4
    gen = markdown.CliTreeMarkdownGenerator(token.tree, cli.root)
    gen.PrintSectionIfExists('DESCRIPTION', disable_header=True)
    doc = gen.Edit()
    fin = io.StringIO(doc)
    lines.extend(render_document.MarkdownRenderer(token_renderer.TokenRenderer(width=width, height=height), fin=fin).Run())
    lines.append([])
    height = 5
    gen = markdown.CliTreeMarkdownGenerator(token.tree, cli.root)
    gen.PrintSynopsisSection()
    doc = gen.Edit()
    fin = io.StringIO(doc)
    lines.extend(render_document.MarkdownRenderer(token_renderer.TokenRenderer(width=width, height=height, compact=False), fin=fin).Run())
    return lines