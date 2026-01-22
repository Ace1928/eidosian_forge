import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
class Settable:
    """Used to configure an attribute to be settable via the set command in the CLI"""

    def __init__(self, name: str, val_type: Union[Type[Any], Callable[[Any], Any]], description: str, settable_object: object, *, settable_attrib_name: Optional[str]=None, onchange_cb: Optional[Callable[[str, _T, _T], Any]]=None, choices: Optional[Iterable[Any]]=None, choices_provider: Optional[ChoicesProviderFunc]=None, completer: Optional[CompleterFunc]=None) -> None:
        """
        Settable Initializer

        :param name: name of the instance attribute being made settable
        :param val_type: callable used to cast the string value from the command line into its proper type and
                         even validate its value. Setting this to bool provides tab completion for true/false and
                         validation using to_bool(). The val_type function should raise an exception if it fails.
                         This exception will be caught and printed by Cmd.do_set().
        :param description: string describing this setting
        :param settable_object: object to which the instance attribute belongs (e.g. self)
        :param settable_attrib_name: name which displays to the user in the output of the set command.
                                     Defaults to `name` if not specified.
        :param onchange_cb: optional function or method to call when the value of this settable is altered
                            by the set command. (e.g. onchange_cb=self.debug_changed)

                            Cmd.do_set() passes the following 3 arguments to onchange_cb:
                                param_name: str - name of the changed parameter
                                old_value: Any - the value before being changed
                                new_value: Any - the value after being changed

        The following optional settings provide tab completion for a parameter's values. They correspond to the
        same settings in argparse-based tab completion. A maximum of one of these should be provided.

        :param choices: iterable of accepted values
        :param choices_provider: function that provides choices for this argument
        :param completer: tab completion function that provides choices for this argument
        """
        if val_type == bool:

            def get_bool_choices(_) -> List[str]:
                """Used to tab complete lowercase boolean values"""
                return ['true', 'false']
            val_type = to_bool
            choices_provider = cast(ChoicesProviderFunc, get_bool_choices)
        self.name = name
        self.val_type = val_type
        self.description = description
        self.settable_obj = settable_object
        self.settable_attrib_name = settable_attrib_name if settable_attrib_name is not None else name
        self.onchange_cb = onchange_cb
        self.choices = choices
        self.choices_provider = choices_provider
        self.completer = completer

    def get_value(self) -> Any:
        """
        Get the value of the settable attribute
        :return:
        """
        return getattr(self.settable_obj, self.settable_attrib_name)

    def set_value(self, value: Any) -> Any:
        """
        Set the settable attribute on the specified destination object
        :param value: New value to set
        :return: New value that the attribute was set to
        """
        new_value = self.val_type(value)
        if self.choices is not None and new_value not in self.choices:
            choices_str = ', '.join(map(repr, self.choices))
            raise ValueError(f'invalid choice: {new_value!r} (choose from {choices_str})')
        orig_value = self.get_value()
        setattr(self.settable_obj, self.settable_attrib_name, new_value)
        if orig_value != new_value and self.onchange_cb:
            self.onchange_cb(self.name, orig_value, new_value)
        return new_value