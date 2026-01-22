import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _mark_set(self, textbuffer, location, mark):
    if mark == self._buffer.get_insert() and self._deferred_check:
        self._check_deferred_range(False)