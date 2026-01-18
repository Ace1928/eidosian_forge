from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
Access the list of children of this directory.

        With a parent_id_basename_to_file_id index, loads all the children,
        without loads the entire index. Without is bad. A more sophisticated
        proxy object might be nice, to allow partial loading of children as
        well when specific names are accessed. (So path traversal can be
        written in the obvious way but not examine siblings.).
        