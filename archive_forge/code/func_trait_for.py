from functools import partial
from .constants import DefaultValue
def trait_for(trait):
    """ Returns the trait corresponding to a specified value.
    """
    from .traits import Trait
    try:
        return as_ctrait(trait)
    except TypeError:
        return Trait(trait)