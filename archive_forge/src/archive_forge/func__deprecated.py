from __future__ import unicode_literals
import re
import warnings
from .. import util
from ..compat import unicode
from ..pattern import RegexPattern
@staticmethod
def _deprecated():
    """
		Warn about deprecation.
		"""
    warnings.warn("GitIgnorePattern ('gitignore') is deprecated. Use GitWildMatchPattern ('gitwildmatch') instead.", DeprecationWarning, stacklevel=3)