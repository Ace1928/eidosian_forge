import copy
import sys
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import Undefined, _validate_everything
from traits.trait_errors import TraitError
class TraitDictObject(TraitDict):
    """ A subclass of TraitDict that fires trait events when mutated.
    This is for backward compatibility with Traits 6.0 and lower.

    This is used by the Dict trait type, and all values set into a Dict
    trait will be copied into a new TraitDictObject instance.

    Mutation of the TraitDictObject will fire a "name_items" event with
    appropriate added, changed and removed values.

    Parameters
    ----------
    trait : CTrait instance
        The CTrait instance associated with the attribute that this dict
        has been set to.
    object : HasTraits instance
        The HasTraits instance that the dict has been set as an attribute for.
    name : str
        The name of the attribute on the object.
    value : dict
        The dict of values to initialize the TraitDictObject with.

    Attributes
    ----------
    trait : CTrait instance
        The CTrait instance associated with the attribute that this dict
        has been set to.
    object : weak reference to a HasTraits instance
        A weak reference to a HasTraits instance that the dict has been set
        as an attribute for.
    name : str
        The name of the attribute on the object.
    name_items : str
        The name of the items event trait that the trait dict will fire when
        mutated.
    """

    def __init__(self, trait, object, name, value):
        self.trait = trait
        self.object = ref(object)
        self.name = name
        self.name_items = None
        if trait.has_items:
            self.name_items = name + '_items'
        super().__init__(value, key_validator=self._key_validator, value_validator=self._value_validator, notifiers=[self.notifier])

    def _key_validator(self, key):
        """ Calls the trait's key_trait.handler.validate.

        Parameters
        ----------
        key : A hashable object.
            The key to validate.

        Returns
        -------
        validated_key : A hashable object.
            The validated key.

        Raises
        ------
        TraitError
            If the validation fails.
        """
        trait = getattr(self, 'trait', None)
        object = getattr(self, 'object', lambda: None)()
        if trait is None or object is None:
            return key
        validate = trait.key_trait.handler.validate
        if validate is None:
            return key
        try:
            return validate(object, self.name, key)
        except TraitError as excep:
            excep.set_prefix('Each key of the')
            raise excep

    def _value_validator(self, value):
        """ Calls the trait's value_handler.validate

        Parameters
        ----------
        value : any
            The value to validate.

        Returns
        -------
        validated_value : any
            The validated value.

        Raises
        ------
        TraitError
            If the validation fails.
        """
        trait = getattr(self, 'trait', None)
        object = getattr(self, 'object', lambda: None)()
        if trait is None or object is None:
            return value
        validate = trait.value_handler.validate
        if validate is None:
            return value
        try:
            return validate(object, self.name, value)
        except TraitError as excep:
            excep.set_prefix('Each value of the')
            raise excep

    def notifier(self, trait_dict, removed, added, changed):
        """ Fire the TraitDictEvent with the provided parameters.

        Parameters
        ----------
        trait_dict : dict
            The complete dictionary.
        removed : dict
            Dict of removed items.
        added : dict
            Dict of added items.
        changed : dict
            Dict of changed items.
        """
        if self.name_items is None:
            return
        object = self.object()
        if object is None:
            return
        if getattr(object, self.name) is not self:
            return
        event = TraitDictEvent(removed=removed, added=added, changed=changed)
        items_event = self.trait.items_event()
        object.trait_items_event(self.name_items, event, items_event)

    def __getstate__(self):
        """ Get the state of the object for serialization.

        Object and trait should not be serialized.
        """
        result = super().__getstate__()
        del result['object']
        del result['trait']
        return result

    def __setstate__(self, state):
        """ Restore the state of the object after serialization.
        """
        state.setdefault('name', '')
        state['notifiers'] = [self.notifier]
        state['object'] = lambda: None
        state['trait'] = None
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        """ Perform a deepcopy operation..

        Object is a weakref and should not be copied.
        """
        result = TraitDictObject(self.trait, lambda: None, self.name, dict((copy.deepcopy(x, memo) for x in self.items())))
        return result