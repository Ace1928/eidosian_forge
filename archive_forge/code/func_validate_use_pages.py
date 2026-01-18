import sys
from collections.abc import MutableSequence
import re
from textwrap import dedent
from keyword import iskeyword
import flask
from ._grouping import grouping_len, map_grouping
from .development.base_component import Component
from . import exceptions
from ._utils import (
def validate_use_pages(config):
    if not config.get('assets_folder', None):
        raise exceptions.PageError('`dash.register_page()` must be called after app instantiation')
    if flask.has_request_context():
        raise exceptions.PageError('\n            dash.register_page() can’t be called within a callback as it updates dash.page_registry, which is a global variable.\n             For more details, see https://dash.plotly.com/sharing-data-between-callbacks#why-global-variables-will-break-your-app\n            ')