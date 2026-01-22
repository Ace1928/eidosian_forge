from textwrap import indent
from typing import Optional, Union
from .typedefs import _CIMultiDict
class PayloadEncodingError(BadHttpMessage):
    """Base class for payload errors"""