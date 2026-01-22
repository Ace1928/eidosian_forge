from __future__ import annotations
import re
import sys
import typing
from collections.abc import MutableMapping, Sequence
from urwid import str_util
import urwid.util  # isort: skip  # pylint: disable=wrong-import-position
class MoreInputRequired(Exception):
    pass