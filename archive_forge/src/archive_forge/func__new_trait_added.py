import re
import string
import weakref
from string import whitespace
from types import MethodType
from .constants import DefaultValue
from .trait_base import Undefined, Uninitialized
from .trait_errors import TraitError
from .trait_notifiers import TraitChangeNotifyWrapper
from .util.weakiddict import WeakIDKeyDict
def _new_trait_added(self, object, name, new_trait):
    """ Handles new traits being added to an object being monitored.
        """
    if new_trait.startswith(self.name[:-1]):
        trait = object.base_trait(new_trait)
        for meta_name, meta_eval in self._metadata.items():
            if not meta_eval(getattr(trait, meta_name)):
                return
        type = SIMPLE_LISTENER
        handler = trait.handler
        if handler is not None:
            type = type_map.get(handler.default_value_, SIMPLE_LISTENER)
        self.active[object].append((new_trait, type))
        getattr(self, type)(object, new_trait, False)