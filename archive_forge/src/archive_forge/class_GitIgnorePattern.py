from __future__ import unicode_literals
import re
import warnings
from .. import util
from ..compat import unicode
from ..pattern import RegexPattern
class GitIgnorePattern(GitWildMatchPattern):
    """
	The :class:`GitIgnorePattern` class is deprecated by :class:`GitWildMatchPattern`.
	This class only exists to maintain compatibility with v0.4.
	"""

    def __init__(self, *args, **kw):
        """
		Warn about deprecation.
		"""
        self._deprecated()
        return super(GitIgnorePattern, self).__init__(*args, **kw)

    @staticmethod
    def _deprecated():
        """
		Warn about deprecation.
		"""
        warnings.warn("GitIgnorePattern ('gitignore') is deprecated. Use GitWildMatchPattern ('gitwildmatch') instead.", DeprecationWarning, stacklevel=3)

    @classmethod
    def pattern_to_regex(cls, *args, **kw):
        """
		Warn about deprecation.
		"""
        cls._deprecated()
        return super(GitIgnorePattern, cls).pattern_to_regex(*args, **kw)