from textwrap import indent
from typing import Optional, Union
from .typedefs import _CIMultiDict
class HttpBadRequest(BadHttpMessage):
    code = 400
    message = 'Bad Request'