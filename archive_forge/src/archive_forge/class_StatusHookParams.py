import sys
from . import delta as _mod_delta
from . import errors as errors
from . import hooks as _mod_hooks
from . import log, osutils
from . import revision as _mod_revision
from . import tsort
from .trace import mutter, warning
from .workingtree import ShelvingUnsupported
class StatusHookParams:
    """Object holding parameters passed to post_status hooks.

    :ivar old_tree: Start tree (basis tree) for comparison.
    :ivar new_tree: Working tree.
    :ivar to_file: If set, write to this file.
    :ivar versioned: Show only versioned files.
    :ivar show_ids: Show internal object ids.
    :ivar short: Use short status indicators.
    :ivar verbose: Verbose flag.
    """

    def __init__(self, old_tree, new_tree, to_file, versioned, show_ids, short, verbose, specific_files=None):
        """Create a group of post_status hook parameters.

        :param old_tree: Start tree (basis tree) for comparison.
        :param new_tree: Working tree.
        :param to_file: If set, write to this file.
        :param versioned: Show only versioned files.
        :param show_ids: Show internal object ids.
        :param short: Use short status indicators.
        :param verbose: Verbose flag.
        :param specific_files: If set, a list of filenames whose status should be
            shown.  It is an error to give a filename that is not in the
            working tree, or in the working inventory or in the basis inventory.
        """
        self.old_tree = old_tree
        self.new_tree = new_tree
        self.to_file = to_file
        self.versioned = versioned
        self.show_ids = show_ids
        self.short = short
        self.verbose = verbose
        self.specific_files = specific_files

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return '<{}({}, {}, {}, {}, {}, {}, {}, {})>'.format(self.__class__.__name__, self.old_tree, self.new_tree, self.to_file, self.versioned, self.show_ids, self.short, self.verbose, self.specific_files)