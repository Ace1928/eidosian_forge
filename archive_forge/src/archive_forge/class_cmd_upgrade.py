import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
class cmd_upgrade(Command):
    __doc__ = 'Upgrade a repository, branch or working tree to a newer format.\n\n    When the default format has changed after a major new release of\n    Bazaar/Breezy, you may be informed during certain operations that you\n    should upgrade. Upgrading to a newer format may improve performance\n    or make new features available. It may however limit interoperability\n    with older repositories or with older versions of Bazaar or Breezy.\n\n    If you wish to upgrade to a particular format rather than the\n    current default, that can be specified using the --format option.\n    As a consequence, you can use the upgrade command this way to\n    "downgrade" to an earlier format, though some conversions are\n    a one way process (e.g. changing from the 1.x default to the\n    2.x default) so downgrading is not always possible.\n\n    A backup.bzr.~#~ directory is created at the start of the conversion\n    process (where # is a number). By default, this is left there on\n    completion. If the conversion fails, delete the new .bzr directory\n    and rename this one back in its place. Use the --clean option to ask\n    for the backup.bzr directory to be removed on successful conversion.\n    Alternatively, you can delete it by hand if everything looks good\n    afterwards.\n\n    If the location given is a shared repository, dependent branches\n    are also converted provided the repository converts successfully.\n    If the conversion of a branch fails, remaining branches are still\n    tried.\n\n    For more information on upgrades, see the Breezy Upgrade Guide,\n    https://www.breezy-vcs.org/doc/en/upgrade-guide/.\n    '
    _see_also = ['check', 'reconcile', 'formats']
    takes_args = ['url?']
    takes_options = [RegistryOption('format', help='Upgrade to a specific format.  See "brz help formats" for details.', lazy_registry=('breezy.controldir', 'format_registry'), converter=lambda name: controldir.format_registry.make_controldir(name), value_switches=True, title='Branch format'), Option('clean', help='Remove the backup.bzr directory if successful.'), Option('dry-run', help="Show what would be done, but don't actually do anything.")]

    def run(self, url='.', format=None, clean=False, dry_run=False):
        from .upgrade import upgrade
        exceptions = upgrade(url, format, clean_up=clean, dry_run=dry_run)
        if exceptions:
            if len(exceptions) == 1:
                raise exceptions[0]
            else:
                return 3