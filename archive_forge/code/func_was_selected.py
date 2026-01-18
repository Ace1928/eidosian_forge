from __future__ import annotations
import argparse
import contextlib
import copy
import enum
import functools
import logging
from typing import Generator
from typing import Sequence
from flake8 import defaults
from flake8 import statistics
from flake8 import utils
from flake8.formatting import base as base_formatter
from flake8.violation import Violation
def was_selected(self, code: str) -> Selected | Ignored:
    """Determine if the code has been selected by the user.

        :param code: The code for the check that has been run.
        :returns:
            Selected.Implicitly if the selected list is empty,
            Selected.Explicitly if the selected list is not empty and a match
            was found,
            Ignored.Implicitly if the selected list is not empty but no match
            was found.
        """
    if code.startswith(self.selected_explicitly):
        return Selected.Explicitly
    elif code.startswith(self.selected):
        return Selected.Implicitly
    else:
        return Ignored.Implicitly