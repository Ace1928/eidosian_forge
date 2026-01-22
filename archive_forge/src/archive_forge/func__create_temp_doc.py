from __future__ import annotations
import logging # isort:skip
import re
from contextlib import contextmanager
from typing import (
from weakref import WeakKeyDictionary
from ..core.types import ID
from ..document.document import Document
from ..model import Model, collect_models
from ..settings import settings
from ..themes.theme import Theme
from ..util.dataclasses import dataclass, field
from ..util.serialization import (
def _create_temp_doc(models: Sequence[Model]) -> Document:
    doc = _new_doc()
    for m in models:
        doc.models[m.id] = m
        m._temp_document = doc
        for ref in m.references():
            doc.models[ref.id] = ref
            ref._temp_document = doc
    doc._roots = list(models)
    return doc