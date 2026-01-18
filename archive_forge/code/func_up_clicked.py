import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
def up_clicked(self, *_):
    self.url.value = self.fs._parent(self.url.value)
    self.go_clicked()