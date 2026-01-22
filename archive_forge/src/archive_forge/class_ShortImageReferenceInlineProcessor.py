from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class ShortImageReferenceInlineProcessor(ImageReferenceInlineProcessor):
    """ Short form of image reference: `![ref]`. """

    def evalId(self, data: str, index: int, text: str) -> tuple[str, int, bool]:
        """Evaluate the id of `[ref]`.  """
        return (text.lower(), index, True)