from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
import six
def Markdown(command, tree):
    """Returns the help markdown document string for the command node in tree.

  Args:
    command: The command node in the root tree.
    tree: The (sub)tree root.

  Returns:
    The markdown document string.
  """
    return CliTreeMarkdownGenerator(command, tree).Generate()