from typing import Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from lxml.html import (
from parsel.selector import create_root_node
from w3lib.html import strip_html5_whitespace
from scrapy.http.request import Request
from scrapy.http.response.text import TextResponse
from scrapy.utils.python import is_listlike, to_bytes
from scrapy.utils.response import get_base_url
def _urlencode(seq: Iterable[FormdataKVType], enc: str) -> str:
    values = [(to_bytes(k, enc), to_bytes(v, enc)) for k, vs in seq for v in (cast(Iterable[str], vs) if is_listlike(vs) else [cast(str, vs)])]
    return urlencode(values, doseq=True)