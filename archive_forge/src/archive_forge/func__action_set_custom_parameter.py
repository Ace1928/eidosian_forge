from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_set_custom_parameter(self: argparse.Action, value: Any) -> None:
    f'\n        Set the custom {param_name} attribute of an argparse Action.\n\n        This function is added by cmd2 as a method called ``{setter_name}()`` to ``argparse.Action`` class.\n\n        To call: ``action.{setter_name}({param_name})``\n\n        :param self: argparse Action being updated\n        :param value: value being assigned\n        '
    if param_type and (not isinstance(value, param_type)):
        raise TypeError(f'{param_name} must be of type {param_type}, got: {value} ({type(value)})')
    setattr(self, attr_name, value)