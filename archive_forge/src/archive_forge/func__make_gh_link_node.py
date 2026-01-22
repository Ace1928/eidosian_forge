from __future__ import annotations
import logging  # isort:skip
from os.path import join
import toml
from docutils import nodes, utils
from docutils.parsers.rst.roles import set_classes
from . import PARALLEL_SAFE
from .util import _REPO_TOP
def _make_gh_link_node(app, rawtext, role, kind, api_type, id, options=None):
    """Return a link to a Bokeh Github resource.

    Args:
        app (Sphinx app) : current app
        rawtext (str) : text being replaced with link node.
        role (str) : role name
        kind (str) : resource type (issue, pull, etc.)
        api_type (str) : type for api link
        id : (str) : id of the resource to link to
        options (dict) : options dictionary passed to role function

    """
    url = f'{BOKEH_GH}/{api_type}/{id}'
    options = options or {}
    set_classes(options)
    node = nodes.reference(rawtext, f'{kind}{utils.unescape(id)}', refuri=url, **options)
    return node