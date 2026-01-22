import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _click_move_popup(self, *args):
    self.move_click_mark(self._buffer.get_iter_at_mark(self._buffer.get_insert()))
    return False