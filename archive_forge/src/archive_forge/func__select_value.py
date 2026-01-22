from typing import Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from lxml.html import (
from parsel.selector import create_root_node
from w3lib.html import strip_html5_whitespace
from scrapy.http.request import Request
from scrapy.http.response.text import TextResponse
from scrapy.utils.python import is_listlike, to_bytes
from scrapy.utils.response import get_base_url
def _select_value(ele: SelectElement, n: Optional[str], v: Union[None, str, MultipleSelectOptions]) -> Tuple[Optional[str], Union[None, str, MultipleSelectOptions]]:
    multiple = ele.multiple
    if v is None and (not multiple):
        o = ele.value_options
        return (n, o[0]) if o else (None, None)
    return (n, v)