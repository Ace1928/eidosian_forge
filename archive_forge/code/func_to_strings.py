import errno
import os
import re
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import commands, errors, option, osutils, registry, trace
def to_strings(self):
    """Generate strings for the provided conflicts"""
    for conflict in self:
        yield str(conflict)