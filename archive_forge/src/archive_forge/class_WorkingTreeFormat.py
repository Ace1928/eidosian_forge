import contextlib
import errno
import os
import sys
from typing import TYPE_CHECKING, Optional, Tuple
import breezy
from .lazy_import import lazy_import
import stat
from breezy import (
from . import errors, mutabletree, osutils
from . import revision as _mod_revision
from .controldir import (ControlComponent, ControlComponentFormat,
from .i18n import gettext
from .symbol_versioning import deprecated_in, deprecated_method
from .trace import mutter, note
from .transport import NoSuchFile
class WorkingTreeFormat(ControlComponentFormat):
    """An encapsulation of the initialization and open routines for a format.

    Formats provide three things:
     * An initialization routine,
     * a format string,
     * an open routine.

    Formats are placed in an dict by their format string for reference
    during workingtree opening. Its not required that these be instances, they
    can be classes themselves with class methods - it simply depends on
    whether state is needed for a given format or not.

    Once a format is deprecated, just deprecate the initialize and open
    methods on the format class. Do not deprecate the object, as the
    object will be created every time regardless.
    """
    requires_rich_root = False
    upgrade_recommended = False
    requires_normalized_unicode_filenames = False
    case_sensitive_filename = 'FoRMaT'
    missing_parent_conflicts = False
    'If this format supports missing parent conflicts.'
    supports_versioned_directories: bool
    supports_merge_modified = True
    'If this format supports storing merge modified hashes.'
    supports_setting_file_ids = True
    'If this format allows setting the file id.'
    supports_store_uncommitted = True
    'If this format supports shelve-like functionality.'
    supports_leftmost_parent_id_as_ghost = True
    supports_righthand_parent_id_as_ghost = True
    ignore_filename: Optional[str] = None
    'Name of file with ignore patterns, if any. '

    def initialize(self, controldir, revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False):
        """Initialize a new working tree in controldir.

        Args:
          controldir: ControlDir to initialize the working tree in.
          revision_id: allows creating a working tree at a different
            revision than the branch is at.
          from_branch: Branch to checkout
          accelerator_tree: A tree which can be used for retrieving file
            contents more quickly than the revision tree, i.e. a workingtree.
            The revision tree will be used for cases where accelerator_tree's
            content is different.
          hardlink: If true, hard-link files from accelerator_tree,
            where possible.
        """
        raise NotImplementedError(self.initialize)

    def __eq__(self, other):
        return self.__class__ is other.__class__

    def __ne__(self, other):
        return not self == other

    def get_format_description(self):
        """Return the short description for this format."""
        raise NotImplementedError(self.get_format_description)

    def is_supported(self):
        """Is this format supported?

        Supported formats can be initialized and opened.
        Unsupported formats may not support initialization or committing or
        some other features depending on the reason for not being supported.
        """
        return True

    def supports_content_filtering(self):
        """True if this format supports content filtering."""
        return False

    def supports_views(self):
        """True if this format supports stored views."""
        return False

    def get_controldir_for_branch(self):
        """Get the control directory format for creating branches.

        This is to support testing of working tree formats that can not exist
        in the same control directory as a branch.
        """
        return self._matchingcontroldir