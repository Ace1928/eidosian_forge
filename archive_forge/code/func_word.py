import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
@property
def word(self):
    start = self.iter
    if not self._iter_worker.starts_word(start):
        self._iter_worker.backward_word_start(start)
    end = self.iter
    if self._iter_worker.inside_word(end):
        self._iter_worker.forward_word_end(end)
    return (start, end)