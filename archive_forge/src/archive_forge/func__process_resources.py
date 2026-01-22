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
def _process_resources(resources: Literal['default'] | None) -> str:
    """ Return an argument to suppress normal Bokeh server resources, if requested.

    Args:
        resources ("default" or None) :
            If None, return an HTML argument to suppress default resources.

    Returns:
        str

    """
    if resources not in ('default', None):
        raise ValueError("`resources` must be either 'default' or None.")
    if resources is None:
        return '&resources=none'
    return ''