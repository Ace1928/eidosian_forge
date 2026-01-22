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
def _get_mutable_section(self):
    """Get the MutableSection for the Stack.

        This is where we guarantee that the mutable section is lazily loaded:
        this means we won't load the corresponding store before setting a value
        or deleting an option. In practice the store will often be loaded but
        this helps catching some programming errors.
        """
    store = self.store
    section = store.get_mutable_section(self.mutable_section_id)
    return (store, section)