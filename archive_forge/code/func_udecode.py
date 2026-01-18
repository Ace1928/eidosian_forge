import codecs
import re
import unicodedata
from abc import ABC, ABCMeta
from typing import Iterator, Tuple, Sequence, Any, NamedTuple
def udecode(self, bytes_: str, final: bool=False) -> str:
    """Decode LaTeX *bytes_* into a unicode string.

        This implementation calls :meth:`get_unicode_tokens` and joins
        the resulting unicode strings together.
        """
    return ''.join(self.get_unicode_tokens(bytes_, final=final))