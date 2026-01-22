from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
def __makePlaceholder(self, type: str) -> tuple[str, str]:
    """ Generate a placeholder """
    id = '%04d' % len(self.stashed_nodes)
    hash = util.INLINE_PLACEHOLDER % id
    return (hash, id)