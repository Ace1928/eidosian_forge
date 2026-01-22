import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
def apply_data_bg(self, value: Any) -> str:
    """
        Apply background color to data text based on what row is being generated and whether a color has been defined
        :param value: object whose text is to be colored
        :return: formatted data string
        """
    if self.row_num % 2 == 0 and self.even_bg is not None:
        return ansi.style(value, bg=self.even_bg)
    elif self.row_num % 2 != 0 and self.odd_bg is not None:
        return ansi.style(value, bg=self.odd_bg)
    else:
        return str(value)