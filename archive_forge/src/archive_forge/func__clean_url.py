from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote_plus, urlparse
from ..core.templates import AUTOLOAD_REQUEST_TAG, FILE
from ..resources import DEFAULT_SERVER_HTTP_URL
from ..util.serialization import make_globally_unique_css_safe_id
from ..util.strings import format_docstring
from .bundle import bundle_for_objs_and_resources
from .elements import html_page_for_render_items
from .util import RenderItem
def _clean_url(url: str) -> str:
    """ Produce a canonical Bokeh server URL.

    Args:
        url (str)
            A URL to clean, or "defatul". If "default" then the
            ``BOKEH_SERVER_HTTP_URL`` will be returned.

    Returns:
        str

    """
    if url == 'default':
        url = DEFAULT_SERVER_HTTP_URL
    if url.startswith('ws'):
        raise ValueError('url should be the http or https URL for the server, not the websocket URL')
    return url.rstrip('/')