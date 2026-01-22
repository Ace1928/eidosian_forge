from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from .element_base import element_base
from .margin import Margin
def _translate_vjust(self, just: float) -> Literal['top', 'bottom', 'center']:
    """
        Translate ggplot2 justification from [0, 1] to top, bottom, center.
        """
    if just == 0:
        return 'bottom'
    elif just == 1:
        return 'top'
    else:
        return 'center'