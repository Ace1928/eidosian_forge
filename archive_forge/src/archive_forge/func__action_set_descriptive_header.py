from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_set_descriptive_header(self: argparse.Action, descriptive_header: Optional[str]) -> None:
    """
    Set the descriptive_header attribute of an argparse Action.

    This function is added by cmd2 as a method called ``set_descriptive_header()`` to ``argparse.Action`` class.

    To call: ``action.set_descriptive_header(descriptive_header)``

    :param self: argparse Action being updated
    :param descriptive_header: value being assigned
    """
    setattr(self, ATTR_DESCRIPTIVE_HEADER, descriptive_header)