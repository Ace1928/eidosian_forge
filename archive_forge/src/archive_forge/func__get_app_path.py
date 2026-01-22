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
def _get_app_path(url: str) -> str:
    """ Extract the app path from a Bokeh server URL

    Args:
        url (str) :

    Returns:
        str

    """
    app_path = urlparse(url).path.rstrip('/')
    if not app_path.startswith('/'):
        app_path = '/' + app_path
    return app_path