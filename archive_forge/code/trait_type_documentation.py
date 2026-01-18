import warnings
from .base_trait_handler import BaseTraitHandler
from .constants import ComparisonMode, DefaultValue, TraitKind
from .trait_base import Missing, Self, TraitsCache, Undefined, class_of
from .trait_dict_object import TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListObject
from .trait_set_object import TraitSetObject
 Instantiate the class an return a CTrait instance

        This is primarily to allow traits to be defined within
        classes without having to explicitly call them.
        