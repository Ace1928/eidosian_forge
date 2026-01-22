from __future__ import annotations
import logging # isort:skip
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from ..core.types import PathLike
from ..document import Document
from ..resources import Resources, ResourcesMode
def _reset_with_doc(self, doc: Document) -> None:
    """ Reset output modes but DO replace the default Document

        """
    self._document = doc
    self._reset_keeping_doc()