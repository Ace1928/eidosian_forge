from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
class _FontStyleRunsRangeIterator:

    def __init__(self, font_names, font_sizes, bolds, italics, stretch, dpi):
        self.zip_iter = runlist.ZipRunIterator((font_names, font_sizes, bolds, italics, stretch))
        self.dpi = dpi

    def ranges(self, start, end):
        from pyglet import font
        for start, end, styles in self.zip_iter.ranges(start, end):
            font_name, font_size, bold, italic, stretch = styles
            ft = font.load(font_name, font_size, bold=bool(bold), italic=bool(italic), stretch=stretch, dpi=self.dpi)
            yield (start, end, ft)

    def __getitem__(self, index):
        from pyglet import font
        font_name, font_size, bold, italic, stretch = self.zip_iter[index]
        return font.load(font_name, font_size, bold=bool(bold), italic=bool(italic), stretch=stretch, dpi=self.dpi)