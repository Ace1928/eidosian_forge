from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import properties
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
class HelpTextGenerator(walker.Walker):
    """Generates help text files in a directory hierarchy.

  Attributes:
    _directory: The help text output directory.
  """

    def __init__(self, cli, directory, hidden=False, progress_callback=None, restrict=None):
        """Constructor.

    Args:
      cli: The Cloud SDK CLI object.
      directory: The Help Text output directory path name.
      hidden: Boolean indicating whether to consider the hidden CLI.
      progress_callback: f(float), The function to call to update the progress
        bar or None for no progress bar.
      restrict: Restricts the walk to the command/group dotted paths in this
        list. For example, restrict=['gcloud.alpha.test', 'gcloud.topic']
        restricts the walk to the 'gcloud topic' and 'gcloud alpha test'
        commands/groups.

    """
        super(HelpTextGenerator, self).__init__(cli, progress_callback=progress_callback, restrict=restrict)
        self._directory = directory
        files.MakeDir(self._directory)

    def Visit(self, node, parent, is_group):
        """Renders a help text doc for each node in the CLI tree.

    Args:
      node: group/command CommandCommon info.
      parent: The parent Visit() return value, None at the top level.
      is_group: True if node is a group, otherwise its is a command.

    Returns:
      The parent value, ignored here.
    """
        command = node.GetPath()
        if is_group:
            directory = os.path.join(self._directory, *command[1:])
        else:
            directory = os.path.join(self._directory, *command[1:-1])
        files.MakeDir(directory, mode=493)
        path = os.path.join(directory, 'GROUP' if is_group else command[-1])
        with files.FileWriter(path) as f:
            md = markdown.Markdown(node)
            render_document.RenderDocument(style='text', fin=io.StringIO(md), out=f)
        return parent