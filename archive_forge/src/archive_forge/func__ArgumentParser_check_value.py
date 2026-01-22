from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _ArgumentParser_check_value(self: argparse.ArgumentParser, action: argparse.Action, value: Any) -> None:
    """
    Custom override of ArgumentParser._check_value that supports CompletionItems as choices.
    When evaluating choices, input is compared to CompletionItem.orig_value instead of the
    CompletionItem instance.

    :param self: ArgumentParser instance
    :param action: the action being populated
    :param value: value from command line already run through conversion function by argparse
    """
    from gettext import gettext as _
    if action.choices is not None:
        choices = [c.orig_value if isinstance(c, CompletionItem) else c for c in action.choices]
        if value not in choices:
            args = {'value': value, 'choices': ', '.join(map(repr, choices))}
            msg = _('invalid choice: %(value)r (choose from %(choices)s)')
            raise ArgumentError(action, msg % args)