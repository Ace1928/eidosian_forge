from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_get_suppress_tab_hint(self: argparse.Action) -> bool:
    """
    Get the suppress_tab_hint attribute of an argparse Action.

    This function is added by cmd2 as a method called ``get_suppress_tab_hint()`` to ``argparse.Action`` class.

    To call: ``action.get_suppress_tab_hint()``

    :param self: argparse Action being queried
    :return: The value of suppress_tab_hint or False if attribute does not exist
    """
    return cast(bool, getattr(self, ATTR_SUPPRESS_TAB_HINT, False))