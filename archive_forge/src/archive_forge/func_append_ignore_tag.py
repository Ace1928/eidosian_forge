import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def append_ignore_tag(self, tag):
    """
        Appends a tag to the list of ignored tags. A string will be automatic
        resolved into a tag object.

        :param tag: Tag object or tag name.
        """
    if isinstance(tag, str):
        tag = self._table.lookup(tag)
    self.ignored_tags.append(tag)