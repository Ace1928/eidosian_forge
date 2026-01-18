from __future__ import annotations
import re
import typing as t
from .._internal import _missing
from ..exceptions import BadRequestKeyError
from .mixins import ImmutableHeadersMixin
from .structures import iter_multi_items
from .structures import MultiDict
from .. import http
def to_wsgi_list(self):
    """Convert the headers into a list suitable for WSGI.

        :return: list
        """
    return list(self)