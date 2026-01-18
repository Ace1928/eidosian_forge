from warnings import warn
from typing import Optional
from w3lib._types import StrOrBytes
def unicode_to_str(text: StrOrBytes, encoding: Optional[str]=None, errors: str='strict') -> bytes:
    warn('The w3lib.utils.unicode_to_str function is deprecated and will be removed in a future release.', DeprecationWarning, stacklevel=2)
    if encoding is None:
        encoding = 'utf-8'
    if isinstance(text, str):
        return text.encode(encoding, errors)
    return text