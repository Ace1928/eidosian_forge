from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
from .._utils import to_rgba
from .._utils.registry import RegistryHierarchyMeta
from ..exceptions import PlotnineError, deprecated_themeable_name
from .elements import element_blank
from .elements.element_base import element_base
class legend_text_colorbar(MixinSequenceOfValues):
    """
    Colorbar text

    Parameters
    ----------
    theme_element : element_text

    Notes
    -----
    Horizontal alignment `ha` has no effect when the text is to the
    left or to the right. Likewise vertical alignment `va` has no
    effect when the text at the top or the bottom.
    """
    _omit = ['margin', 'ha', 'va']

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        super().apply_figure(figure, targets)
        if (texts := targets.legend_text_colorbar):
            self.set(texts)

    def blank_figure(self, figure: Figure, targets: ThemeTargets):
        super().blank_figure(figure, targets)
        if (texts := targets.legend_text_colorbar):
            for text in texts:
                text.set_visible(False)