from typing import Union
from .align import AlignMethod
from .cells import cell_len, set_cell_size
from .console import Console, ConsoleOptions, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .style import Style
from .text import Text
def _rule_line(self, chars_len: int, width: int) -> Text:
    rule_text = Text(self.characters * (width // chars_len + 1), self.style)
    rule_text.truncate(width)
    rule_text.plain = set_cell_size(rule_text.plain, width)
    return rule_text