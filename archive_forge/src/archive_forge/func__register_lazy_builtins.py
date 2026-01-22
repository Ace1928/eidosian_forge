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
def _register_lazy_builtins():
    for name, aliases, module_name in [('cmd_bisect', [], 'breezy.bisect'), ('cmd_bundle_info', [], 'breezy.bzr.bundle.commands'), ('cmd_config', [], 'breezy.config'), ('cmd_dump_btree', [], 'breezy.bzr.debug_commands'), ('cmd_file_id', [], 'breezy.bzr.debug_commands'), ('cmd_file_path', [], 'breezy.bzr.debug_commands'), ('cmd_version_info', [], 'breezy.cmd_version_info'), ('cmd_resolve', ['resolved'], 'breezy.conflicts'), ('cmd_conflicts', [], 'breezy.conflicts'), ('cmd_ping', [], 'breezy.bzr.smart.ping'), ('cmd_sign_my_commits', [], 'breezy.commit_signature_commands'), ('cmd_verify_signatures', [], 'breezy.commit_signature_commands'), ('cmd_test_script', [], 'breezy.cmd_test_script')]:
        builtin_command_registry.register_lazy(name, aliases, module_name)