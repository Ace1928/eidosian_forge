import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
class Char:
    """Class encapsulating a single character, its foreground, background and style attributes."""
    __slots__ = ('data', 'fg', 'bg', 'bold', 'italics', 'underscore', 'blink', 'strikethrough', 'reverse')

    def __init__(self, data=' ', fg=ANSI_FG_DEFAULT, bg=ANSI_BG_DEFAULT, bold=False, italics=False, underscore=False, blink=False, strikethrough=False, reverse=False):
        self.data = data
        self.fg = fg
        self.bg = bg
        self.bold = bold
        self.italics = italics
        self.underscore = underscore
        self.blink = blink
        self.strikethrough = strikethrough
        self.reverse = reverse

    def reset(self):
        default = self.__class__()
        for k in self.__slots__[1:]:
            self[k] = default[k]

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def copy(self, **kwargs):
        attrs = {}
        for k in self.__slots__:
            if k in kwargs:
                attrs[k] = kwargs[k]
            else:
                attrs[k] = self[k]
        return self.__class__(**attrs)

    def __eq__(self, other):
        for k in self.__slots__:
            if self[k] != other[k]:
                return False
        return True