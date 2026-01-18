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
def tree_files_for_add(file_list):
    """
    Return a tree and list of absolute paths from a file list.

    Similar to tree_files, but add handles files a bit differently, so it a
    custom implementation.  In particular, MutableTreeTree.smart_add expects
    absolute paths, which it immediately converts to relative paths.
    """
    if file_list:
        tree, relpath = WorkingTree.open_containing(file_list[0])
        if tree.supports_views():
            view_files = tree.views.lookup_view()
            if view_files:
                for filename in file_list:
                    if not osutils.is_inside_any(view_files, filename):
                        raise views.FileOutsideView(filename, view_files)
        file_list = file_list[:]
        file_list[0] = tree.abspath(relpath)
    else:
        tree = WorkingTree.open_containing('.')[0]
        if tree.supports_views():
            view_files = tree.views.lookup_view()
            if view_files:
                file_list = view_files
                view_str = views.view_display_str(view_files)
                note(gettext('Ignoring files outside view. View is %s'), view_str)
    return (tree, file_list)