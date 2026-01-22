from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_get_descriptive_header(self: argparse.Action) -> Optional[str]:
    """
    Get the descriptive_header attribute of an argparse Action.

    This function is added by cmd2 as a method called ``get_descriptive_header()`` to ``argparse.Action`` class.

    To call: ``action.get_descriptive_header()``

    :param self: argparse Action being queried
    :return: The value of descriptive_header or None if attribute does not exist
    """
    return cast(Optional[str], getattr(self, ATTR_DESCRIPTIVE_HEADER, None))