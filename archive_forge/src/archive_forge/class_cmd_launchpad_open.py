from ... import branch as _mod_branch
from ... import controldir, trace
from ...commands import Command
from ...errors import CommandError, NotBranchError
from ...i18n import gettext
from ...option import ListOption, Option
class cmd_launchpad_open(Command):
    __doc__ = 'Open a Launchpad branch page in your web browser.'
    aliases = ['lp-open']
    takes_options = [Option('dry-run', 'Do not actually open the browser. Just say the URL we would use.')]
    takes_args = ['location?']

    def run(self, location=None, dry_run=False):
        trace.warning('lp-open is deprecated. Please use web-open instead')
        from breezy.plugins.propose.cmds import cmd_web_open
        return cmd_web_open().run(location=location, dry_run=dry_run)