from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
class CanBeStopped(Protocol):

    def stop(self) -> None:
        ...