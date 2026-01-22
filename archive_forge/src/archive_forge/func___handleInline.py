from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
def __handleInline(self, data: str, patternIndex: int=0) -> str:
    """
        Process string with inline patterns and replace it with placeholders.

        Arguments:
            data: A line of Markdown text.
            patternIndex: The index of the `inlinePattern` to start with.

        Returns:
            String with placeholders.

        """
    if not isinstance(data, util.AtomicString):
        startIndex = 0
        count = len(self.inlinePatterns)
        while patternIndex < count:
            data, matched, startIndex = self.__applyPattern(self.inlinePatterns[patternIndex], data, patternIndex, startIndex)
            if not matched:
                patternIndex += 1
    return data