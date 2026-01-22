import warnings
from typing import Any, Dict, Iterable, List, Optional, Set  # noqa
from yarl import URL
from .typedefs import LooseHeaders, StrOrURL
from .web_response import Response
class HTTPSuccessful(HTTPException):
    """Base class for exceptions with status codes in the 200s."""