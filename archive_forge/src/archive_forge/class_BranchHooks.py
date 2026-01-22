from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
class BranchHooks(Hooks):
    """A dictionary mapping hook name to a list of callables for branch hooks.

    e.g. ['post_push'] Is the list of items to be called when the
    push function is invoked.
    """

    def __init__(self):
        """Create the default hooks.

        These are all empty initially, because by default nothing should get
        notified.
        """
        Hooks.__init__(self, 'breezy.branch', 'Branch.hooks')
        self.add_hook('open', 'Called with the Branch object that has been opened after a branch is opened.', (1, 8))
        self.add_hook('post_push', 'Called after a push operation completes. post_push is called with a breezy.branch.BranchPushResult object and only runs in the bzr client.', (0, 15))
        self.add_hook('post_pull', 'Called after a pull operation completes. post_pull is called with a breezy.branch.PullResult object and only runs in the bzr client.', (0, 15))
        self.add_hook('pre_commit', 'Called after a commit is calculated but before it is completed. pre_commit is called with (local, master, old_revno, old_revid, future_revno, future_revid, tree_delta, future_tree). old_revid is NULL_REVISION for the first commit to a branch, tree_delta is a TreeDelta object describing changes from the basis revision. hooks MUST NOT modify this delta.  future_tree is an in-memory tree obtained from CommitBuilder.revision_tree() and hooks MUST NOT modify this tree.', (0, 91))
        self.add_hook('post_commit', 'Called in the bzr client after a commit has completed. post_commit is called with (local, master, old_revno, old_revid, new_revno, new_revid). old_revid is NULL_REVISION for the first commit to a branch.', (0, 15))
        self.add_hook('post_uncommit', 'Called in the bzr client after an uncommit completes. post_uncommit is called with (local, master, old_revno, old_revid, new_revno, new_revid) where local is the local branch or None, master is the target branch, and an empty branch receives new_revno of 0, new_revid of None.', (0, 15))
        self.add_hook('pre_change_branch_tip', 'Called in bzr client and server before a change to the tip of a branch is made. pre_change_branch_tip is called with a breezy.branch.ChangeBranchTipParams. Note that push, pull, commit, uncommit will all trigger this hook.', (1, 6))
        self.add_hook('post_change_branch_tip', 'Called in bzr client and server after a change to the tip of a branch is made. post_change_branch_tip is called with a breezy.branch.ChangeBranchTipParams. Note that push, pull, commit, uncommit will all trigger this hook.', (1, 4))
        self.add_hook('transform_fallback_location', 'Called when a stacked branch is activating its fallback locations. transform_fallback_location is called with (branch, url), and should return a new url. Returning the same url allows it to be used as-is, returning a different one can be used to cause the branch to stack on a closer copy of that fallback_location. Note that the branch cannot have history accessing methods called on it during this hook because the fallback locations have not been activated. When there are multiple hooks installed for transform_fallback_location, all are called with the url returned from the previous hook.The order is however undefined.', (1, 9))
        self.add_hook('automatic_tag_name', 'Called to determine an automatic tag name for a revision. automatic_tag_name is called with (branch, revision_id) and should return a tag name or None if no tag name could be determined. The first non-None tag name returned will be used.', (2, 2))
        self.add_hook('post_branch_init', 'Called after new branch initialization completes. post_branch_init is called with a breezy.branch.BranchInitHookParams. Note that init, branch and checkout (both heavyweight and lightweight) will all trigger this hook.', (2, 2))
        self.add_hook('post_switch', 'Called after a checkout switches branch. post_switch is called with a breezy.branch.SwitchHookParams.', (2, 2))