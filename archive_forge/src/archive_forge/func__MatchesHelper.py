from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
def _MatchesHelper(self, pattern_parts, path):
    """Determines whether the given pattern matches the given path.

    Args:
      pattern_parts: list of str, the list of pattern parts that must all match
        the path.
      path: str, the path to match.

    Returns:
      bool, whether the patch matches the pattern_parts (Matches() will convert
        this into a Match value).
    """
    if not pattern_parts:
        return True
    if path is None:
        return False
    pattern_part = pattern_parts[-1]
    remaining_pattern = pattern_parts[:-1]
    if path:
        path = os.path.normpath(path)
    remaining_path, path_part = os.path.split(path)
    if not path_part:
        remaining_path = None
    if pattern_part == '**':
        path_prefixes = GetPathPrefixes(path)
        if not (remaining_pattern and remaining_pattern[0] == ''):
            remaining_pattern.insert(0, '')
        return any((self._MatchesHelper(remaining_pattern, prefix) for prefix in path_prefixes))
    if pattern_part == '*' and (not remaining_pattern):
        if remaining_path and len(remaining_path) > 1:
            return False
    if not fnmatch.fnmatch(path_part, pattern_part):
        return False
    return self._MatchesHelper(remaining_pattern, remaining_path)