from __future__ import annotations
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from . import util
import re
@util.deprecated('This class is deprecated and will be removed in the future; use [`UnescapeTreeprocessor`][markdown.treeprocessors.UnescapeTreeprocessor] instead.')
class UnescapePostprocessor(Postprocessor):
    """ Restore escaped chars. """
    RE = re.compile('{}(\\d+){}'.format(util.STX, util.ETX))

    def unescape(self, m: re.Match[str]) -> str:
        return chr(int(m.group(1)))

    def run(self, text: str) -> str:
        return self.RE.sub(self.unescape, text)