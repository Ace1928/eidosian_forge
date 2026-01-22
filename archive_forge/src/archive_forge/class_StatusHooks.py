import sys
from . import delta as _mod_delta
from . import errors as errors
from . import hooks as _mod_hooks
from . import log, osutils
from . import revision as _mod_revision
from . import tsort
from .trace import mutter, warning
from .workingtree import ShelvingUnsupported
class StatusHooks(_mod_hooks.Hooks):
    """A dictionary mapping hook name to a list of callables for status hooks.

    e.g. ['post_status'] Is the list of items to be called when the
    status command has finished printing the status.
    """

    def __init__(self):
        """Create the default hooks.

        These are all empty initially, because by default nothing should get
        notified.
        """
        _mod_hooks.Hooks.__init__(self, 'breezy.status', 'hooks')
        self.add_hook('post_status', 'Called with argument StatusHookParams after Breezy has displayed the status. StatusHookParams has the attributes (old_tree, new_tree, to_file, versioned, show_ids, short, verbose). The last four arguments correspond to the command line options specified by the user for the status command. to_file is the output stream for writing.', (2, 3))
        self.add_hook('pre_status', 'Called with argument StatusHookParams before Breezy displays the status. StatusHookParams has the attributes (old_tree, new_tree, to_file, versioned, show_ids, short, verbose). The last four arguments correspond to the command line options specified by the user for the status command. to_file is the output stream for writing.', (2, 3))