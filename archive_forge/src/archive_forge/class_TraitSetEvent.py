import copy
import copyreg
from itertools import chain
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
class TraitSetEvent(object):
    """ An object reporting in-place changes to a traits sets.

    Parameters
    ----------
    removed : set, optional
        Old values that were removed from the set.
    added : set, optional
        New values added to the set.

    Attributes
    ----------
    removed : set
        Old values that were removed from the set.
    added : set
        New values added to the set.
    """

    def __init__(self, *, removed=None, added=None):
        if removed is None:
            removed = set()
        self.removed = removed
        if added is None:
            added = set()
        self.added = added

    def __repr__(self):
        return f'{self.__class__.__name__}(removed={self.removed!r}, added={self.added!r})'