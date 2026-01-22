from textwrap import indent
from typing import Optional, Union
from .typedefs import _CIMultiDict
class ContentEncodingError(PayloadEncodingError):
    """Content encoding error."""