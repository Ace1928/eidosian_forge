from importlib import import_module
import sys
from types import FunctionType, MethodType
from .constants import DefaultValue, ValidateTrait
from .trait_base import (
from .trait_base import RangeTypes  # noqa: F401, used by TraitsUI
from .trait_errors import TraitError
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_converters import trait_from
from .trait_handler import TraitHandler
from .trait_list_object import TraitListEvent, TraitListObject
from .util.deprecated import deprecated
class TraitPrefixList(TraitHandler):
    """Ensures that a value assigned to a trait attribute is a member of a
    list of specified string values, or is a unique prefix of one of those
    values.

    TraitPrefixList is a variation on TraitEnum. The values that can be
    assigned to a trait attribute defined using a TraitPrefixList handler is
    the set of all strings supplied to the TraitPrefixList constructor, as well
    as any unique prefix of those strings. That is, if the set of strings
    supplied to the constructor is described by
    [*s*\\ :sub:`1`\\ , *s*\\ :sub:`2`\\ , ..., *s*\\ :sub:`n`\\ ], then the string
    *v* is a valid value for the trait if *v* == *s*\\ :sub:`i[:j]` for one and
    only one pair of values (i, j). If *v* is a valid value, then the actual
    value assigned to the trait attribute is the corresponding *s*\\ :sub:`i`
    value that *v* matched.

    As with TraitEnum, the list of legal values can be provided as a list
    or tuple of values.  That is, ``TraitPrefixList(['one', 'two', 'three'])``
    and ``TraitPrefixList('one', 'two', 'three')`` are equivalent.

    Example
    -------
    ::

        class Person(HasTraits):
            married = Trait('no', TraitPrefixList('yes', 'no')

    The Person class has a **married** trait that accepts any of the
    strings 'y', 'ye', 'yes', 'n', or 'no' as valid values. However, the actual
    values assigned as the value of the trait attribute are limited to either
    'yes' or 'no'. That is, if the value 'y' is assigned to the **married**
    attribute, the actual value assigned will be 'yes'.

    Note that the algorithm used by TraitPrefixList in determining whether a
    string is a valid value is fairly efficient in terms of both time and
    space, and is not based on a brute force set of comparisons.

    Parameters
    ----------
    *values
        Either all legal string values for the enumeration, or a single list
        or tuple of legal string values.

    Attributes
    ----------
    values : tuple of strings
        Enumeration of all legal values for a trait.
    """

    @deprecated(_WARNING_FORMAT_STR.format(handler='TraitPrefixList', replacement='PrefixList'))
    def __init__(self, *values):
        if len(values) == 1 and type(values[0]) in SequenceTypes:
            values = values[0]
        self.values = values[:]
        self.values_ = values_ = {}
        for key in values:
            values_[key] = key
        self.fast_validate = (ValidateTrait.prefix_map, values_, self.validate)

    def validate(self, object, name, value):
        try:
            if value not in self.values_:
                match = None
                n = len(value)
                for key in self.values:
                    if value == key[:n]:
                        if match is not None:
                            match = None
                            break
                        match = key
                if match is None:
                    self.error(object, name, value)
                self.values_[value] = match
            return self.values_[value]
        except:
            self.error(object, name, value)

    def info(self):
        return ' or '.join([repr(x) for x in self.values]) + ' (or any unique prefix)'

    def get_editor(self, trait):
        from traitsui.api import EnumEditor
        return EnumEditor(values=self, cols=trait.cols or 3)

    def __getstate__(self):
        result = self.__dict__.copy()
        if 'fast_validate' in result:
            del result['fast_validate']
        return result