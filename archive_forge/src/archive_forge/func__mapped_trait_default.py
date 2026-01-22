from functools import partial
from .constants import DefaultValue
def _mapped_trait_default(trait, name, instance):
    """ Callable providing default for a shadow trait of a mapped trait pair.

    Parameters
    ----------
    trait : CTrait
        The principal trait of the mapped trait pair.
    name : str
        The name of the trait on the relevant HasTraits object.
    instance : HasTraits
        The HasTraits object on which the mapped trait lives.

    Returns
    -------
    default : object
        The default value for the shadow trait.
    """
    value = getattr(instance, name)
    return trait.handler.mapped_value(value)