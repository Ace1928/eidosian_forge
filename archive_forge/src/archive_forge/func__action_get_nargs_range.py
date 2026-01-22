from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_get_nargs_range(self: argparse.Action) -> Optional[Tuple[int, Union[int, float]]]:
    """
    Get the nargs_range attribute of an argparse Action.

    This function is added by cmd2 as a method called ``get_nargs_range()`` to ``argparse.Action`` class.

    To call: ``action.get_nargs_range()``

    :param self: argparse Action being queried
    :return: The value of nargs_range or None if attribute does not exist
    """
    return cast(Optional[Tuple[int, Union[int, float]]], getattr(self, ATTR_NARGS_RANGE, None))