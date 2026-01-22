import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _get_languages_menu(self):
    if _IS_GTK3:
        return self._build_languages_menu()
    else:
        if self._languages_menu is None:
            self._languages_menu = self._build_languages_menu()
        return self._languages_menu