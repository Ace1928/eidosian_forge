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
class ToolbarButton(Button):
    """ A Button trait type whose UI editor is a toolbar button.

    This is just a Button trait with different defaults to style it like
    a toolbar button.

    Parameters
    ----------
    label : str
        The label for the button.
    image : pyface.ImageResource
        An image to display on the button.
    style : 'button', 'radio', 'toolbar' or 'checkbox'
        The style of button to display.
    orientation : 'horizontal' or 'vertical'
        The orientation of the label relative to the image.
    width_padding : integer between 0 and 31
        Extra padding (in pixels) added to the left and right sides of
        the button.
    height_padding : integer between 0 and 31
        Extra padding (in pixels) added to the top and bottom of the
        button.
    **metadata
        Trait metadata for the trait.

    Attributes
    ----------
    label : str
        The label for the button.
    image : pyface.ImageResource
        An image to display on the button.
    style : 'button', 'radio', 'toolbar' or 'checkbox'
        The style of button to display.
    values_trait : str
        For a "button" or "toolbar" style, the name of an enum
        trait whose values will populate a drop-down menu on the button.
        The selected value will replace the label on the button.
    orientation : 'horizontal' or 'vertical'
        The orientation of the label relative to the image.
    width_padding : integer between 0 and 31
        Extra padding (in pixels) added to the left and right sides of
        the button.
    height_padding : integer between 0 and 31
        Extra padding (in pixels) added to the top and bottom of the
        button.
    view : traitsui View, optional
        An optional View to display when the button is clicked.
    """

    def __init__(self, label='', image=None, style='toolbar', orientation='vertical', width_padding=2, height_padding=2, **metadata):
        super().__init__(label, image=image, style=style, orientation=orientation, width_padding=width_padding, height_padding=height_padding, **metadata)