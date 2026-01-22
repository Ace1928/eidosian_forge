from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_set_completer(self: argparse.Action, completer: CompleterFunc) -> None:
    """
    Set completer of an argparse Action.

    This function is added by cmd2 as a method called ``set_completer()`` to ``argparse.Action`` class.

    To call: ``action.set_completer(completer)``

    :param self: action being edited
    :param completer: the completer instance to use
    :raises: TypeError if used on incompatible action type
    """
    self._set_choices_callable(ChoicesCallable(is_completer=True, to_call=completer))