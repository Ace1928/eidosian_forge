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
def _process_relative_urls(relative_urls: bool, url: str) -> str:
    """ Return an absolute URL HTML argument to add to a Bokeh server URL, if
    requested.

    Args:
        relative_urls (book) :
            If false, generate an absolute URL to add.

        url (str) :
            The absolute URL to add as an HTML argument

    Returns:
        str

    """
    if relative_urls:
        return ''
    return '&bokeh-absolute-url=' + url