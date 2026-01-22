import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class SectionMatcher:
    """Select sections into a given Store.

    This is intended to be used to postpone getting an iterable of sections
    from a store.
    """

    def __init__(self, store):
        self.store = store

    def get_sections(self):
        sections = self.store.get_sections()
        for store, s in sections:
            if self.match(s):
                yield (store, s)

    def match(self, section):
        """Does the proposed section match.

        Args:
          section: A Section object.

        Returns:
          True if the section matches, False otherwise.
        """
        raise NotImplementedError(self.match)