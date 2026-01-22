from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def _iter_elements(elements, length):
    last = 0
    for element in elements:
        p = element.position
        yield (last, p, None)
        yield (p, p + 1, element)
        last = p + 1
    yield (last, length, None)