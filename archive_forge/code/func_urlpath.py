import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
@property
def urlpath(self):
    """URL of currently selected item"""
    return f'{self.protocol.value}://{self.main.value[0]}' if self.main.value else None