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
class cmd_pack(Command):
    __doc__ = 'Compress the data within a repository.\n\n    This operation compresses the data within a bazaar repository. As\n    bazaar supports automatic packing of repository, this operation is\n    normally not required to be done manually.\n\n    During the pack operation, bazaar takes a backup of existing repository\n    data, i.e. pack files. This backup is eventually removed by bazaar\n    automatically when it is safe to do so. To save disk space by removing\n    the backed up pack files, the --clean-obsolete-packs option may be\n    used.\n\n    Warning: If you use --clean-obsolete-packs and your machine crashes\n    during or immediately after repacking, you may be left with a state\n    where the deletion has been written to disk but the new packs have not\n    been. In this case the repository may be unusable.\n    '
    _see_also = ['repositories']
    takes_args = ['branch_or_repo?']
    takes_options = [Option('clean-obsolete-packs', 'Delete obsolete packs to save disk space.')]

    def run(self, branch_or_repo='.', clean_obsolete_packs=False):
        dir = controldir.ControlDir.open_containing(branch_or_repo)[0]
        try:
            branch = dir.open_branch()
            repository = branch.repository
        except errors.NotBranchError:
            repository = dir.open_repository()
        repository.pack(clean_obsolete_packs=clean_obsolete_packs)