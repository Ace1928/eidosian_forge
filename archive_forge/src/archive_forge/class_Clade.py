import collections
import copy
import itertools
import random
import re
import warnings
class Clade(TreeElement, TreeMixin):
    """A recursively defined sub-tree.

    :Parameters:
        branch_length : str
            The length of the branch leading to the root node of this clade.
        name : str
            The clade's name (a label).
        clades : list
            Sub-trees rooted directly under this tree's root.
        confidence : number
            Support.
        color : BranchColor
            The display color of the branch and descendents.
        width : number
            The display width of the branch and descendents.

    """

    def __init__(self, branch_length=None, name=None, clades=None, confidence=None, color=None, width=None):
        """Define parameters for the Clade tree."""
        self.branch_length = branch_length
        self.name = name
        self.clades = clades or []
        self.confidence = confidence
        self.color = color
        self.width = width

    @property
    def root(self):
        """Allow TreeMixin methods to traverse clades properly."""
        return self

    def is_terminal(self):
        """Check if this is a terminal (leaf) node."""
        return not self.clades

    def __getitem__(self, index):
        """Get clades by index (integer or slice)."""
        if isinstance(index, (int, slice)):
            return self.clades[index]
        ref = self
        for idx in index:
            ref = ref[idx]
        return ref

    def __iter__(self):
        """Iterate through this tree's direct descendent clades (sub-trees)."""
        return iter(self.clades)

    def __len__(self):
        """Return the number of clades directly under the root."""
        return len(self.clades)

    def __bool__(self):
        """Boolean value of an instance of this class (True).

        NB: If this method is not defined, but ``__len__``  is, then the object
        is considered true if the result of ``__len__()`` is nonzero. We want
        Clade instances to always be considered True.
        """
        return True

    def __str__(self) -> str:
        """Return name of the class instance."""
        if self.name:
            return self.name[:37] + '...' if len(self.name) > 40 else self.name
        return self.__class__.__name__

    def _get_color(self):
        return self._color

    def _set_color(self, arg):
        if arg is None or isinstance(arg, BranchColor):
            self._color = arg
        elif isinstance(arg, str):
            if arg in BranchColor.color_names:
                self._color = BranchColor.from_name(arg)
            elif arg.startswith('#') and len(arg) == 7:
                self._color = BranchColor.from_hex(arg)
            else:
                raise ValueError(f'invalid color string {arg}')
        elif hasattr(arg, '__iter__') and len(arg) == 3:
            self._color = BranchColor(*arg)
        else:
            raise ValueError(f'invalid color value {arg}')
    color = property(_get_color, _set_color, doc='Branch color.')