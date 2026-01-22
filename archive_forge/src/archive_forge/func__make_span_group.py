import warnings
import weakref
from collections import UserDict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import srsly
from ..errors import Errors, Warnings
from .span_group import SpanGroup
def _make_span_group(self, name: str, spans: Iterable['Span']) -> SpanGroup:
    doc = self._ensure_doc()
    return SpanGroup(doc, name=name, spans=spans)