import copy
import operator
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import class_of, Undefined, _validate_everything
from traits.trait_errors import TraitError
def _item_validator(self, value):
    """
        Validate an item that's being added to the list.
        """
    object = self.object()
    if object is None:
        return value
    trait_validator = self.trait.item_trait.handler.validate
    if trait_validator is None:
        return value
    try:
        return trait_validator(object, self.name, value)
    except TraitError as excp:
        excp.set_prefix('Each element of the')
        raise