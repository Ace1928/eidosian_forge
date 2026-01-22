from typing import Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from lxml.html import (
from parsel.selector import create_root_node
from w3lib.html import strip_html5_whitespace
from scrapy.http.request import Request
from scrapy.http.response.text import TextResponse
from scrapy.utils.python import is_listlike, to_bytes
from scrapy.utils.response import get_base_url
def _get_clickable(clickdata: Optional[dict], form: FormElement) -> Optional[Tuple[str, str]]:
    """
    Returns the clickable element specified in clickdata,
    if the latter is given. If not, it returns the first
    clickable element found
    """
    clickables = list(form.xpath('descendant::input[re:test(@type, "^(submit|image)$", "i")]|descendant::button[not(@type) or re:test(@type, "^submit$", "i")]', namespaces={'re': 'http://exslt.org/regular-expressions'}))
    if not clickables:
        return None
    if clickdata is None:
        el = clickables[0]
        return (el.get('name'), el.get('value') or '')
    nr = clickdata.get('nr', None)
    if nr is not None:
        try:
            el = list(form.inputs)[nr]
        except IndexError:
            pass
        else:
            return (el.get('name'), el.get('value') or '')
    xpath = './/*' + ''.join((f'[@{k}="{v}"]' for k, v in clickdata.items()))
    el = form.xpath(xpath)
    if len(el) == 1:
        return (el[0].get('name'), el[0].get('value') or '')
    if len(el) > 1:
        raise ValueError(f'Multiple elements found ({el!r}) matching the criteria in clickdata: {clickdata!r}')
    else:
        raise ValueError(f'No clickable element matching clickdata: {clickdata!r}')