import codecs
import re
import unicodedata
from abc import ABC, ABCMeta
from typing import Iterator, Tuple, Sequence, Any, NamedTuple
def uencode(self, unicode_: str, final: bool=False) -> str:
    """Encode the *unicode_* string into LaTeX :class:`bytes`.

        This implementation calls :meth:`get_latex_chars` and joins
        the resulting :class:`bytes` together.
        """
    return ''.join(self.get_latex_chars(unicode_, final=final))