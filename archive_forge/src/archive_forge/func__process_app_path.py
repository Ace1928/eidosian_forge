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
def _process_app_path(app_path: str) -> str:
    """ Return an app path HTML argument to add to a Bokeh server URL.

    Args:
        app_path (str) :
            The app path to add. If the app path is ``/`` then it will be
            ignored and an empty string returned.

    """
    if app_path == '/':
        return ''
    return '&bokeh-app-path=' + app_path