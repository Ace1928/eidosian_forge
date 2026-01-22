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
class GlobalStack(Stack):
    """Global options only stack.

    The following sections are queried:

    * command-line overrides,

    * the 'DEFAULT' section in bazaar.conf

    This stack will use the ``DEFAULT`` section in bazaar.conf as its
    MutableSection.
    """

    def __init__(self):
        gstore = self.get_shared_store(GlobalStore())
        super().__init__([self._get_overrides, NameMatcher(gstore, 'DEFAULT').get_sections], gstore, mutable_section_id='DEFAULT')