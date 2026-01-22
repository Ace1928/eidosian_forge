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
class HtmlGenerator(DocumentGenerator):
    """Generates HTML manpage files with suffix .html in an output directory.

  The output directory will contain a man1 subdirectory containing all of the
  HTML manpage files.
  """

    def WriteHtmlMenu(self, command, out):
        """Writes the command menu tree HTML on out.

    Args:
      command: dict, The tree (nested dict) of command/group names.
      out: stream, The output stream.
    """

        def ConvertPathToIdentifier(path):
            return '_'.join(path)

        def WalkCommandTree(command, prefix):
            """Visit each command and group in the CLI command tree.

      Args:
        command: dict, The tree (nested dict) of command/group names.
        prefix: [str], The subcommand arg prefix.
      """
            level = len(prefix)
            visibility = 'visible' if level <= 1 else 'hidden'
            indent = level * 2 + 2
            name = command.get('_name_')
            args = prefix + [name]
            out.write('{indent}<li class="{visibility}" id="{item}" onclick="select(event, this.id)">{name}'.format(indent=' ' * indent, visibility=visibility, name=name, item=ConvertPathToIdentifier(args)))
            commands = command.get('commands', []) + command.get('groups', [])
            if commands:
                out.write('<ul>\n')
                for c in sorted(commands, key=lambda x: x['_name_']):
                    WalkCommandTree(c, args)
                out.write('{indent}</ul>\n'.format(indent=' ' * (indent + 1)))
                out.write('{indent}</li>\n'.format(indent=' ' * indent))
            else:
                out.write('</li>\n'.format(indent=' ' * (indent + 1)))
        out.write('<html>\n<head>\n<meta name="description" content="man page tree navigation">\n<meta name="generator" content="gcloud meta generate-help-docs --html-dir=.">\n<title> man page tree navigation </title>\n<base href="." target="_blank">\n<link rel="stylesheet" type="text/css" href="_menu_.css">\n<script type="text/javascript" src="_menu_.js"></script>\n</head>\n<body>\n\n<div class="menu">\n <ul>\n')
        WalkCommandTree(command, [])
        out.write(' </ul>\n</div>\n\n</body>\n</html>\n')

    def _GenerateHtmlNav(self, directory, cli, hidden, restrict):
        """Generates html nav files in directory."""
        tree = CommandTreeGenerator(cli).Walk(hidden, restrict)
        with files.FileWriter(os.path.join(directory, '_menu_.html')) as out:
            self.WriteHtmlMenu(tree, out)
        for file_name in _HELP_HTML_DATA_FILES:
            file_contents = pkg_resources.GetResource('googlecloudsdk.api_lib.meta.help_html_data.', file_name)
            files.WriteBinaryFileContents(os.path.join(directory, file_name), file_contents)

    def __init__(self, cli, directory, hidden=False, progress_callback=None, restrict=None):
        """Constructor.

    Args:
      cli: The Cloud SDK CLI object.
      directory: The HTML output directory path name.
      hidden: Boolean indicating whether to consider the hidden CLI.
      progress_callback: f(float), The function to call to update the progress
        bar or None for no progress bar.
      restrict: Restricts the walk to the command/group dotted paths in this
        list. For example, restrict=['gcloud.alpha.test', 'gcloud.topic']
        restricts the walk to the 'gcloud topic' and 'gcloud alpha test'
        commands/groups.

    """
        super(HtmlGenerator, self).__init__(cli, directory=directory, style='html', suffix='.html')
        self._GenerateHtmlNav(directory, cli, hidden, restrict)