from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def _insert_text(self, start, text, attributes):
    super()._insert_text(start, text, attributes)
    len_text = len(text)
    for runs in self._style_runs.values():
        runs.insert(start, len_text)
    if attributes is not None:
        for attribute, value in attributes.items():
            try:
                runs = self._style_runs[attribute]
            except KeyError:
                runs = self._style_runs[attribute] = runlist.RunList(0, None)
                runs.insert(0, len(self.text))
            runs.set_run(start, start + len_text, value)