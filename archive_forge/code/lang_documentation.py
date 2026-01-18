import re
import collections
from . import _compat, tools
Return copy of ``s`` that will not treat ``'<...>'`` as DOT HTML string in quoting.

    Args:
        s: String in which leading ``'<'`` and trailing ``'>'`` should be treated as literal.
    Raises:
        TypeError: If ``s`` is not a ``str`` on Python 3, or a ``str``/``unicode`` on Python 2.

    >>> quote('<>-*-<>')
    '<>-*-<>'

    >>> quote(nohtml('<>-*-<>'))
    '"<>-*-<>"'
    