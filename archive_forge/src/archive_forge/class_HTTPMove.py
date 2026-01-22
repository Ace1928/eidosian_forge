import warnings
from typing import Any, Dict, Iterable, List, Optional, Set  # noqa
from yarl import URL
from .typedefs import LooseHeaders, StrOrURL
from .web_response import Response
class HTTPMove(HTTPRedirection):

    def __init__(self, location: StrOrURL, *, headers: Optional[LooseHeaders]=None, reason: Optional[str]=None, body: Any=None, text: Optional[str]=None, content_type: Optional[str]=None) -> None:
        if not location:
            raise ValueError('HTTP redirects need a location to redirect to.')
        super().__init__(headers=headers, reason=reason, body=body, text=text, content_type=content_type)
        self.headers['Location'] = str(URL(location))
        self.location = location