import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def add_to_dictionary(self, word):
    """
        Adds a word to user's dictionary.

        :param word: The word to add.
        """
    self._dictionary.add_to_pwl(word)
    self.recheck()