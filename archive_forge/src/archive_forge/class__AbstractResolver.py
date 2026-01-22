import abc
from typing import Any
class _AbstractResolver(_with_metaclass(abc.ABCMeta)):
    """
    This class exists only for documentation purposes to explain how to create a resolver.

    Some examples on how to resolve things:
    - list: get_dictionary could return a dict with index->item and use the index to resolve it later
    - set: get_dictionary could return a dict with id(object)->object and reiterate in that array to resolve it later
    - arbitrary instance: get_dictionary could return dict with attr_name->attr and use getattr to resolve it later
    """

    @abc.abstractmethod
    def resolve(self, var, attribute):
        """
        In this method, we'll resolve some child item given the string representation of the item in the key
        representing the previously asked dictionary.

        :param var: this is the actual variable to be resolved.
        :param attribute: this is the string representation of a key previously returned in get_dictionary.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dictionary(self, var):
        """
        :param var: this is the variable that should have its children gotten.

        :return: a dictionary where each pair key, value should be shown to the user as children items
        in the variables view for the given var.
        """
        raise NotImplementedError