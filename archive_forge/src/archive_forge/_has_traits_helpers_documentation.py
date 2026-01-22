from traits.constants import ComparisonMode, TraitKind
from traits.ctraits import CHasTraits
from traits.observation._observe import add_or_remove_notifiers
from traits.observation.exceptions import NotifierNotFound
from traits.trait_base import Undefined, Uninitialized
 Return true if the CTrait change event should be skipped.

    Parameters
    ----------
    event : TraitChangeEvent

    Returns
    -------
    skipped : bool
        Whether the event should be skipped
    