import sys
from . import delta as _mod_delta
from . import errors as errors
from . import hooks as _mod_hooks
from . import log, osutils
from . import revision as _mod_revision
from . import tsort
from .trace import mutter, warning
from .workingtree import ShelvingUnsupported
Create a group of post_status hook parameters.

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
        