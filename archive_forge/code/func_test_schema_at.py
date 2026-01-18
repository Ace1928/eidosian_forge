import copy
import re
import types
from .ucre import build_re
def test_schema_at(self, text, name, position):
    """Similar to :meth:`linkify_it.main.LinkifyIt.test` but checks only
        specific protocol tail exactly at given position.

        Args:
            text (str): text to scan
            name (str): rule (schema) name
            position (int): length of found pattern (0 on fail).

        Returns:
            int: text (str): text to search
        """
    if not self._compiled.get(name.lower()):
        return 0
    return self._compiled.get(name.lower()).get('validate')(text, position)