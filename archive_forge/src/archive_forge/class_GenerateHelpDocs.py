from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import walker_util
from googlecloudsdk.command_lib.meta import help_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
class GenerateHelpDocs(base.Command):
    """Generate and/or update help document directories.

  The DevSite docs are generated in the --devsite-dir directory with pathnames
  in the reference directory hierarchy. The manpage docs are generated in the
  --manpage-dir directory with pathnames in the manN/ directory hierarchy.
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('--hidden', action='store_true', default=False, help='Include documents for hidden commands and groups.')
        parser.add_argument('--devsite-dir', metavar='DIRECTORY', help='The directory where the generated DevSite reference document subtree will be written. If not specified then DevSite documents will not be generated.')
        parser.add_argument('--help-text-dir', metavar='DIRECTORY', help='The directory where the generated help text reference document subtree will be written. If not specified then help text documents will not be generated. The --hidden flag is implied for --help-text-dir.')
        parser.add_argument('--html-dir', metavar='DIRECTORY', help='The directory where the standalone manpage HTML files will be generated. index.html contains manpage tree navigation in the left pane. The active command branch and its immediate children are visible and clickable. Hover to navigate the tree. Run `python -m http.server 8888 &` in DIRECTORY and point your browser at [](http://localhost:8888) to view the manpage tree. If not specified then the HTML manpage site will not be generated.')
        parser.add_argument('--linter-dir', metavar='DIRECTORY', help='The directory where the generated documentation linter errors for the help text reference document subtree will be written. If not specified then documentation linter documents will not be generated.')
        parser.add_argument('--manpage-dir', metavar='DIRECTORY', help='The directory where the generated manpage document subtree will be written. The manpage hierarchy is flat with all command documents in the manN/ subdirectory. If not specified then manpage documents will not be generated.')
        parser.add_argument('--test', action='store_true', help='Show but do not apply --update actions. Exit with non-zero exit status if any help document file must be updated.')
        parser.add_argument('--update', action='store_true', default=False, help='Update destination directories to match the current CLI. Documents for commands not present in the current CLI will be deleted. Use this flag to update the help text golden files after the help_text_test test fails.')
        parser.add_argument('--update-help-text-dir', hidden=True, metavar='DIRECTORY', help='Deprecated. Use --update --help-text-dir=DIRECTORY instead.')
        parser.add_argument('restrict', metavar='COMMAND/GROUP', nargs='*', default=None, help='Restrict document generation to these dotted command paths.\n              For example:\n\n                gcloud.alpha gcloud.beta.test\n\n              OR\n\n                gcloud.{alpha.,beta.,}compute.instances\n              ')

    def Run(self, args):
        out_of_date = set()

        def Generate(kind, generator, directory, encoding='utf-8', hidden=False):
            """Runs generator and optionally updates help docs in directory."""
            restrict_dir = [re.sub('_', '-', p) for p in args.restrict]
            console_attr.ResetConsoleAttr(encoding)
            if not args.update:
                generator(self._cli_power_users_only, directory).Walk(hidden, restrict_dir)
            elif help_util.HelpUpdater(self._cli_power_users_only, directory, generator, test=args.test, hidden=hidden).Update(restrict_dir):
                out_of_date.add(kind)
        if args.update_help_text_dir:
            log.warning('[--update-help-text-dir={directory}] is deprecated. Use this instead: --update --help-text-dir={directory}.'.format(directory=args.update_help_text_dir))
            args.help_text_dir = args.update_help_text_dir
            args.update = True
        if args.devsite_dir:
            Generate('DevSite', walker_util.DevSiteGenerator, args.devsite_dir, hidden=args.hidden)
        if args.help_text_dir:
            Generate('help text', walker_util.HelpTextGenerator, args.help_text_dir, 'ascii', hidden=True)
        if args.html_dir:
            Generate('html', walker_util.HtmlGenerator, args.html_dir, hidden=args.hidden)
        if args.manpage_dir:
            Generate('man page', walker_util.ManPageGenerator, args.manpage_dir, hidden=args.hidden)
        if args.linter_dir:
            Generate('command linter', walker_util.LinterGenerator, args.linter_dir, hidden=args.hidden)
        if out_of_date and args.test:
            names = sorted(out_of_date)
            if len(names) > 1:
                kinds = ' and '.join([', '.join(names[:-1]), names[-1]])
            else:
                kinds = names[0]
            raise HelpOutOfDateError('{} document files must be updated.'.format(kinds))