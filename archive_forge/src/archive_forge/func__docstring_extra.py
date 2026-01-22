from __future__ import annotations
import logging # isort:skip
from inspect import Parameter
from ..models import Marker
def _docstring_extra(extra_docs):
    return '' if extra_docs is None else extra_docs