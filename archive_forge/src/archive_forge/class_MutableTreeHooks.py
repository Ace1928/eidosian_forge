from typing import List, Optional, Union
from . import errors, hooks, osutils, trace, tree
class MutableTreeHooks(hooks.Hooks):
    """A dictionary mapping a hook name to a list of callables for mutabletree
    hooks.
    """

    def __init__(self):
        """Create the default hooks.

        """
        hooks.Hooks.__init__(self, 'breezy.mutabletree', 'MutableTree.hooks')
        self.add_hook('start_commit', 'Called before a commit is performed on a tree. The start commit hook is able to change the tree before the commit takes place. start_commit is called with the breezy.mutabletree.MutableTree that the commit is being performed on.', (1, 4))
        self.add_hook('post_commit', 'Called after a commit is performed on a tree. The hook is called with a breezy.mutabletree.PostCommitHookParams object. The mutable tree the commit was performed on is available via the mutable_tree attribute of that object.', (2, 0))
        self.add_hook('pre_transform', 'Called before a tree transform on this tree. The hook is called with the tree that is being transformed and the transform.', (2, 5))
        self.add_hook('post_build_tree', 'Called after a completely new tree is built. The hook is called with the tree as its only argument.', (2, 5))
        self.add_hook('post_transform', 'Called after a tree transform has been performed on a tree. The hook is called with the tree that is being transformed and the transform.', (2, 5))