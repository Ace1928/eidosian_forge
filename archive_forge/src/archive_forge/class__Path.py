import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
class _Path(list):
    """List type to use as __path__ but containing additional details.

    Python 3 allows any iterable for __path__ but Python 2 is more fussy.
    """

    def __init__(self, package_name, blocked, extra, paths):
        super().__init__(paths)
        self.package_name = package_name
        self.blocked_names = blocked
        self.extra_details = extra

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {})'.format(self.__class__.__name__, self.package_name, self.blocked_names, self.extra_details, list.__repr__(self))