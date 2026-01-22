import collections.abc
import datetime
from importlib import import_module
import operator
from os import fspath
from os.path import isfile, isdir
import re
import sys
from types import FunctionType, MethodType, ModuleType
import uuid
import warnings
from .constants import DefaultValue, TraitKind, ValidateTrait
from .trait_base import (
from .trait_converters import trait_from, trait_cast
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListEvent, TraitListObject
from .trait_set_object import TraitSetEvent, TraitSetObject
from .trait_type import (
from .traits import (
from .util.deprecated import deprecated
from .util.import_symbol import import_symbol
from .editor_factories import (
class BaseDirectory(BaseStr):
    """ A trait type whose value must be a directory path string.

    This also accepts objects implementing the :class:`os.PathLike` interface,
    converting them to the corresponding string.

    Parameters
    ----------
    value : str
        The default value for the trait.
    auto_set : bool
        Indicates whether the directory editor updates the trait value
        after every key stroke.
    entries : int
        A hint to the TraitsUI editor about how many values to display in
        the editor.
    exists : bool
        Indicates whether the trait value must be an existing directory or
        not.

    Attributes
    ----------
    auto_set : bool
        Indicates whether the directory editor updates the trait value
        after every key stroke.
    entries : int
        A hint to the TraitsUI editor about how many values to display in
        the editor.
    exists : bool
        Indicates whether the trait value must be an existing directory or
        not.
    """

    def __init__(self, value='', auto_set=False, entries=0, exists=False, **metadata):
        self.entries = entries
        self.auto_set = auto_set
        self.exists = exists
        super().__init__(value, **metadata)

    def validate(self, object, name, value):
        """ Validates that a specified value is valid for this trait.

        Note: The 'fast validator' version performs this check in C.
        """
        try:
            value = fspath(value)
        except TypeError:
            pass
        validated_value = super().validate(object, name, value)
        if not self.exists:
            return validated_value
        elif isdir(value):
            return validated_value
        self.error(object, name, value)

    def info(self):
        """ Return a description of the type of value this trait accepts. """
        description = 'a string or os.PathLike object'
        if self.exists:
            description += ' referring to an existing directory'
        return description

    def create_editor(self):
        from traitsui.editors.directory_editor import DirectoryEditor
        editor = DirectoryEditor(auto_set=self.auto_set, entries=self.entries)
        return editor