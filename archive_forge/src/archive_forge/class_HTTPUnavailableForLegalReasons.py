import warnings
from typing import Any, Dict, Iterable, List, Optional, Set  # noqa
from yarl import URL
from .typedefs import LooseHeaders, StrOrURL
from .web_response import Response
class HTTPUnavailableForLegalReasons(HTTPClientError):
    status_code = 451

    def __init__(self, link: Optional[StrOrURL], *, headers: Optional[LooseHeaders]=None, reason: Optional[str]=None, body: Any=None, text: Optional[str]=None, content_type: Optional[str]=None) -> None:
        super().__init__(headers=headers, reason=reason, body=body, text=text, content_type=content_type)
        self._link = None
        if link:
            self._link = URL(link)
            self.headers['Link'] = f'<{str(self._link)}>; rel="blocked-by"'

    @property
    def link(self) -> Optional[URL]:
        return self._link