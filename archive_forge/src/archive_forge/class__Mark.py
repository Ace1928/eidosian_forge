import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
class _Mark:

    def __init__(self, buffer, name, start, iter_worker):
        self._buffer = buffer
        self._name = name
        self._mark = self._buffer.create_mark(self._name, start, True)
        self._iter_worker = iter_worker

    @property
    def iter(self):
        return self._buffer.get_iter_at_mark(self._mark)

    @property
    def inside_word(self):
        return self._iter_worker.inside_word(self.iter)

    @property
    def word(self):
        start = self.iter
        if not self._iter_worker.starts_word(start):
            self._iter_worker.backward_word_start(start)
        end = self.iter
        if self._iter_worker.inside_word(end):
            self._iter_worker.forward_word_end(end)
        return (start, end)

    def move(self, location):
        self._buffer.move_mark(self._mark, location)