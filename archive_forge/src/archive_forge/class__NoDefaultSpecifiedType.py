import warnings
from .base_trait_handler import BaseTraitHandler
from .constants import ComparisonMode, DefaultValue, TraitKind
from .trait_base import Missing, Self, TraitsCache, Undefined, class_of
from .trait_dict_object import TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListObject
from .trait_set_object import TraitSetObject
class _NoDefaultSpecifiedType(object):
    """
    An instance of this class is used to provide the singleton object
    ``NoDefaultSpecified`` for use in the TraitType constructor.
    """