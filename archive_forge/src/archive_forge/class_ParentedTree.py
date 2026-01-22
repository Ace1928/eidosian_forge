import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
class ParentedTree(AbstractParentedTree):
    """
    A ``Tree`` that automatically maintains parent pointers for
    single-parented trees.  The following are methods for querying
    the structure of a parented tree: ``parent``, ``parent_index``,
    ``left_sibling``, ``right_sibling``, ``root``, ``treeposition``.

    Each ``ParentedTree`` may have at most one parent.  In
    particular, subtrees may not be shared.  Any attempt to reuse a
    single ``ParentedTree`` as a child of more than one parent (or
    as multiple children of the same parent) will cause a
    ``ValueError`` exception to be raised.

    ``ParentedTrees`` should never be used in the same tree as ``Trees``
    or ``MultiParentedTrees``.  Mixing tree implementations may result
    in incorrect parent pointers and in ``TypeError`` exceptions.
    """

    def __init__(self, node, children=None):
        self._parent = None
        'The parent of this Tree, or None if it has no parent.'
        super().__init__(node, children)
        if children is None:
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    child._parent = None
                    self._setparent(child, i)

    def _frozen_class(self):
        from nltk.tree.immutable import ImmutableParentedTree
        return ImmutableParentedTree

    def copy(self, deep=False):
        if not deep:
            warnings.warn(f'{self.__class__.__name__} objects do not support shallow copies. Defaulting to a deep copy.')
        return super().copy(deep=True)

    def parent(self):
        """The parent of this tree, or None if it has no parent."""
        return self._parent

    def parent_index(self):
        """
        The index of this tree in its parent.  I.e.,
        ``ptree.parent()[ptree.parent_index()] is ptree``.  Note that
        ``ptree.parent_index()`` is not necessarily equal to
        ``ptree.parent.index(ptree)``, since the ``index()`` method
        returns the first child that is equal to its argument.
        """
        if self._parent is None:
            return None
        for i, child in enumerate(self._parent):
            if child is self:
                return i
        assert False, 'expected to find self in self._parent!'

    def left_sibling(self):
        """The left sibling of this tree, or None if it has none."""
        parent_index = self.parent_index()
        if self._parent and parent_index > 0:
            return self._parent[parent_index - 1]
        return None

    def right_sibling(self):
        """The right sibling of this tree, or None if it has none."""
        parent_index = self.parent_index()
        if self._parent and parent_index < len(self._parent) - 1:
            return self._parent[parent_index + 1]
        return None

    def root(self):
        """
        The root of this tree.  I.e., the unique ancestor of this tree
        whose parent is None.  If ``ptree.parent()`` is None, then
        ``ptree`` is its own root.
        """
        root = self
        while root.parent() is not None:
            root = root.parent()
        return root

    def treeposition(self):
        """
        The tree position of this tree, relative to the root of the
        tree.  I.e., ``ptree.root[ptree.treeposition] is ptree``.
        """
        if self.parent() is None:
            return ()
        else:
            return self.parent().treeposition() + (self.parent_index(),)

    def _delparent(self, child, index):
        assert isinstance(child, ParentedTree)
        assert self[index] is child
        assert child._parent is self
        child._parent = None

    def _setparent(self, child, index, dry_run=False):
        if not isinstance(child, ParentedTree):
            raise TypeError('Can not insert a non-ParentedTree into a ParentedTree')
        if hasattr(child, '_parent') and child._parent is not None:
            raise ValueError('Can not insert a subtree that already has a parent.')
        if not dry_run:
            child._parent = self