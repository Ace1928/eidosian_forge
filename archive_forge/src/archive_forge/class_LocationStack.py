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
class LocationStack(Stack):
    """Per-location options falling back to global options stack.


    The following sections are queried:

    * command-line overrides,

    * the sections matching ``location`` in ``locations.conf``, the order being
      defined by the number of path components in the section glob, higher
      numbers first (from most specific section to most generic).

    * the 'DEFAULT' section in bazaar.conf

    This stack will use the ``location`` section in locations.conf as its
    MutableSection.
    """

    def __init__(self, location):
        """Make a new stack for a location and global configuration.

        Args:
          location: A URL prefix to """
        lstore = self.get_shared_store(LocationStore())
        if location.startswith('file://'):
            location = urlutils.local_path_from_url(location)
        gstore = self.get_shared_store(GlobalStore())
        super().__init__([self._get_overrides, LocationMatcher(lstore, location).get_sections, NameMatcher(gstore, 'DEFAULT').get_sections], lstore, mutable_section_id=location)