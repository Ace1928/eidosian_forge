from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
class _ElementIterator(runlist.RunIterator):

    def __init__(self, elements, length):
        self._run_list_iter = _iter_elements(elements, length)
        self.start, self.end, self.value = next(self)