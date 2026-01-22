import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def _update_files(delta, files, stop_on):
    """Update the set of files to search based on file lifecycle events.

    :param files: a set of files to update
    :param stop_on: either 'add' or 'remove' - take files out of the
      files set once their add or remove entry is detected respectively
    """
    if stop_on == 'add':
        for item in delta.added:
            if item.path[1] in files:
                files.remove(item.path[1])
        for item in delta.copied + delta.renamed:
            if item.path[1] in files:
                files.remove(item.path[1])
                files.add(item.path[0])
            if item.kind[1] == 'directory':
                for path in list(files):
                    if is_inside(item.path[1], path):
                        files.remove(path)
                        files.add(item.path[0] + path[len(item.path[1]):])
    elif stop_on == 'delete':
        for item in delta.removed:
            if item.path[0] in files:
                files.remove(item.path[0])
        for item in delta.copied + delta.renamed:
            if item.path[0] in files:
                files.remove(item.path[0])
                files.add(item.path[1])
            if item.kind[0] == 'directory':
                for path in list(files):
                    if is_inside(item.path[0], path):
                        files.remove(path)
                        files.add(item.path[1] + path[len(item.path[0]):])