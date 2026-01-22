import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
class AbstractParentedTree(Tree, metaclass=ABCMeta):
    """
    An abstract base class for a ``Tree`` that automatically maintains
    pointers to parent nodes.  These parent pointers are updated
    whenever any change is made to a tree's structure.  Two subclasses
    are currently defined:

      - ``ParentedTree`` is used for tree structures where each subtree
        has at most one parent.  This class should be used in cases
        where there is no"sharing" of subtrees.

      - ``MultiParentedTree`` is used for tree structures where a
        subtree may have zero or more parents.  This class should be
        used in cases where subtrees may be shared.

    Subclassing
    ===========
    The ``AbstractParentedTree`` class redefines all operations that
    modify a tree's structure to call two methods, which are used by
    subclasses to update parent information:

      - ``_setparent()`` is called whenever a new child is added.
      - ``_delparent()`` is called whenever a child is removed.
    """

    def __init__(self, node, children=None):
        super().__init__(node, children)
        if children is not None:
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    self._setparent(child, i, dry_run=True)
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    self._setparent(child, i)

    @abstractmethod
    def _setparent(self, child, index, dry_run=False):
        """
        Update the parent pointer of ``child`` to point to ``self``.  This
        method is only called if the type of ``child`` is ``Tree``;
        i.e., it is not called when adding a leaf to a tree.  This method
        is always called before the child is actually added to the
        child list of ``self``.

        :type child: Tree
        :type index: int
        :param index: The index of ``child`` in ``self``.
        :raise TypeError: If ``child`` is a tree with an impropriate
            type.  Typically, if ``child`` is a tree, then its type needs
            to match the type of ``self``.  This prevents mixing of
            different tree types (single-parented, multi-parented, and
            non-parented).
        :param dry_run: If true, the don't actually set the child's
            parent pointer; just check for any error conditions, and
            raise an exception if one is found.
        """

    @abstractmethod
    def _delparent(self, child, index):
        """
        Update the parent pointer of ``child`` to not point to self.  This
        method is only called if the type of ``child`` is ``Tree``; i.e., it
        is not called when removing a leaf from a tree.  This method
        is always called before the child is actually removed from the
        child list of ``self``.

        :type child: Tree
        :type index: int
        :param index: The index of ``child`` in ``self``.
        """

    def __delitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = slice_bounds(self, index, allow_step=True)
            for i in range(start, stop, step):
                if isinstance(self[i], Tree):
                    self._delparent(self[i], i)
            super().__delitem__(index)
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError('index out of range')
            if isinstance(self[index], Tree):
                self._delparent(self[index], index)
            super().__delitem__(index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError('The tree position () may not be deleted.')
            elif len(index) == 1:
                del self[index[0]]
            else:
                del self[index[0]][index[1:]]
        else:
            raise TypeError('%s indices must be integers, not %s' % (type(self).__name__, type(index).__name__))

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            start, stop, step = slice_bounds(self, index, allow_step=True)
            if not isinstance(value, (list, tuple)):
                value = list(value)
            for i, child in enumerate(value):
                if isinstance(child, Tree):
                    self._setparent(child, start + i * step, dry_run=True)
            for i in range(start, stop, step):
                if isinstance(self[i], Tree):
                    self._delparent(self[i], i)
            for i, child in enumerate(value):
                if isinstance(child, Tree):
                    self._setparent(child, start + i * step)
            super().__setitem__(index, value)
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError('index out of range')
            if value is self[index]:
                return
            if isinstance(value, Tree):
                self._setparent(value, index)
            if isinstance(self[index], Tree):
                self._delparent(self[index], index)
            super().__setitem__(index, value)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError('The tree position () may not be assigned to.')
            elif len(index) == 1:
                self[index[0]] = value
            else:
                self[index[0]][index[1:]] = value
        else:
            raise TypeError('%s indices must be integers, not %s' % (type(self).__name__, type(index).__name__))

    def append(self, child):
        if isinstance(child, Tree):
            self._setparent(child, len(self))
        super().append(child)

    def extend(self, children):
        for child in children:
            if isinstance(child, Tree):
                self._setparent(child, len(self))
            super().append(child)

    def insert(self, index, child):
        if index < 0:
            index += len(self)
        if index < 0:
            index = 0
        if isinstance(child, Tree):
            self._setparent(child, index)
        super().insert(index, child)

    def pop(self, index=-1):
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError('index out of range')
        if isinstance(self[index], Tree):
            self._delparent(self[index], index)
        return super().pop(index)

    def remove(self, child):
        index = self.index(child)
        if isinstance(self[index], Tree):
            self._delparent(self[index], index)
        super().remove(child)
    if hasattr(list, '__getslice__'):

        def __getslice__(self, start, stop):
            return self.__getitem__(slice(max(0, start), max(0, stop)))

        def __delslice__(self, start, stop):
            return self.__delitem__(slice(max(0, start), max(0, stop)))

        def __setslice__(self, start, stop, value):
            return self.__setitem__(slice(max(0, start), max(0, stop)), value)

    def __getnewargs__(self):
        """Method used by the pickle module when un-pickling.
        This method provides the arguments passed to ``__new__``
        upon un-pickling. Without this method, ParentedTree instances
        cannot be pickled and unpickled in Python 3.7+ onwards.

        :return: Tuple of arguments for ``__new__``, i.e. the label
            and the children of this node.
        :rtype: Tuple[Any, List[AbstractParentedTree]]
        """
        return (self._label, list(self))