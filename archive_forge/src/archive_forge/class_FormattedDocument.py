from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
class FormattedDocument(AbstractDocument):
    """Simple implementation of a document that maintains text formatting.

    Changes to text style are applied according to the description in
    :py:class:`~pyglet.text.document.AbstractDocument`.  All styles default to ``None``.
    """

    def __init__(self, text=''):
        self._style_runs = {}
        super().__init__(text)

    def get_style_runs(self, attribute):
        try:
            return self._style_runs[attribute].get_run_iterator()
        except KeyError:
            return _no_style_range_iterator

    def get_style(self, attribute, position=0):
        try:
            return self._style_runs[attribute][position]
        except KeyError:
            return None

    def _set_style(self, start, end, attributes):
        for attribute, value in attributes.items():
            try:
                runs = self._style_runs[attribute]
            except KeyError:
                runs = self._style_runs[attribute] = runlist.RunList(0, None)
                runs.insert(0, len(self._text))
            runs.set_run(start, end, value)

    def get_font_runs(self, dpi=None):
        return _FontStyleRunsRangeIterator(self.get_style_runs('font_name'), self.get_style_runs('font_size'), self.get_style_runs('bold'), self.get_style_runs('italic'), self.get_style_runs('stretch'), dpi)

    def get_font(self, position, dpi=None):
        runs_iter = self.get_font_runs(dpi)
        return runs_iter[position]

    def get_element_runs(self):
        return _ElementIterator(self._elements, len(self._text))

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

    def _delete_text(self, start, end):
        super()._delete_text(start, end)
        for runs in self._style_runs.values():
            runs.delete(start, end)