from textwrap import indent
from typing import Optional, Union
from .typedefs import _CIMultiDict
class TransferEncodingError(PayloadEncodingError):
    """transfer encoding error."""