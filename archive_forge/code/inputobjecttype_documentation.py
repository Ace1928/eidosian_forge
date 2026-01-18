from typing import TYPE_CHECKING
from .base import BaseOptions, BaseType
from .inputfield import InputField
from .unmountedtype import UnmountedType
from .utils import yank_fields_from_attrs

        This function is called when the unmounted type (InputObjectType instance)
        is mounted (as a Field, InputField or Argument)
        