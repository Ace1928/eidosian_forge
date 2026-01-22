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
class cmd_hooks(Command):
    __doc__ = 'Show hooks.'
    hidden = True

    def run(self):
        for hook_key in sorted(hooks.known_hooks.keys()):
            some_hooks = hooks.known_hooks_key_to_object(hook_key)
            self.outf.write('%s:\n' % type(some_hooks).__name__)
            for hook_name, hook_point in sorted(some_hooks.items()):
                self.outf.write('  {}:\n'.format(hook_name))
                found_hooks = list(hook_point)
                if found_hooks:
                    for hook in found_hooks:
                        self.outf.write('    %s\n' % (some_hooks.get_hook_name(hook),))
                else:
                    self.outf.write(gettext('    <no hooks installed>\n'))